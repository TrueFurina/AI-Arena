#!/usr/bin/env python3
"""P2-B B-1 训练与评测入口：限量欺骗 · 零和自对弈（设计见 DESIGN-P2B-MARL.md）。

协议（§4 交替冻结自对弈，不用同步联合更新——非平稳环境下两个 Q 表互相
追逐不收敛到可解释结论）：
  Phase H  主持预训练：提问者冻结为 DistinctRandomPolicy，训练 DeceptiveHost
  Phase Q  对抗训练：主持冻结（ε=0.1 随机化，防提问者过拟合单一确定性
            策略），训练提问者
  Phase C  共演 R 轮（可选）：交替 [主持 vs 冻结提问者(评估态)] →
            [提问者 vs 冻结主持(ε=0.1)]，输出共演轨迹诊断（是否震荡）

评测（§5 矩阵，held-out 题目为主、训练集副）：
  提问者 {Q-honest（仅对诚实主持训练=当前 L2 口径）, Q-adv（对抗训练产物）}
  × 主持 {truthful, random-liar（同配额均匀随机，E1' 关键对照）,
          first-k-liar（退化对照）, trained-deceptive}
  指标：tdhit（主读数，但 E1' 判据不用它——见下）/ dhit（表观）/ F1 /
  reward / 复问率 / 谎言数

判据（§6 预注册 + §6-A 预实验修订，先定后跑，不为凑正结果改判据）：
  E1'  训练主持每谎 **F1** 伤害 > random-liar（同 λ 同 k）——主判据。
       原 E1 用 tdhit 伤害，但提问者不复问时那是恒等式 1/max_rounds
       （§6-A 发现 1：每谎恰好把一个本会命中的首答变成非命中，与位置
       无关；sanity 实测复问率恒 0），无法区分任何主持，已弃用并存档
       对照。复问区间的吸收边界见 tests 与 §6-A。
  E1'' dhit 每谎伤害为辅证，照实报。
  E2   Q-adv 在 trained-deceptive 下的 tdhit > Q-honest 同环境（主判据，
       预期可能结构性趋平——§6-A 发现 3：复问被谎类别不如问新类别）；
       E2-f1 F1 鲁棒增益作次要读数。
  E3   混合对局（mock 提问 × MARL 主持）端到端跑通并产出指标。

λ 扫描（§6-A 发现 2）：默认 λ=0.05 实测主持恒真（谎言收益 2×ΔF1 < λ），
处理不激活；故 λ 作为扫描参数，每个 λ 独立全流程，「主持是否说谎」本身
作为区间图报告。E1'/E2 只在主持实际说谎的 λ 区间判定。

执行纪律（项目 2026-10-13 起）：训练/评测一律上服务器 172.16.222.99
（纯 Python 零依赖，python3 直跑；单 seed 分钟级）。本机只跑轻量单测。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.game import TurtleSoupGame
from engine.puzzles import load_puzzles
from engine.evaluate import distinct_hit_rate, metrics_from_result, tdhit
from engine.semantic import (
    CATEGORY_NAMES, CategoryHost, category_answer, parse_category,
)
from eval_generalization import DistinctRandomPolicy, split_puzzles
from players.host_rl import DeceptiveHost, FirstKLiarHost, RandomLiarHost
from players.mock import MockJudge, MockQuestioner
from players.rl import RLQuestioner, TabularQPolicy, run_episode

EPS_HOST = 0.15        # 主持训练探索率
EPS_FROZEN = 0.1       # Phase Q/C 冻结主持的随机化（防过拟合单一确定性策略）
K_DEFAULT = 2          # 每局谎言配额（§3.1）
LAMBDA_SWEEP = (0.0, 0.005, 0.01, 0.02, 0.05)  # §6-A 发现 2：λ 扫描区间


# ---------------- 装配辅助 ----------------
def make_questioner(policy, explore):
    return RLQuestioner(policy=policy, action_bank=list(CATEGORY_NAMES),
                        explore=explore, use_puzzle_state=False)


def make_truth_fn(puzzle):
    def tf(question):
        cat = parse_category(question)
        return None if cat is None else category_answer(puzzle.solution, cat)
    return tf


def _keyword_truth_for(puzzle):
    """(puzzle, question) → golden_qa 子串匹配答案（DeceptiveHost.truth_fn 用）。

    E3 混合对局注入：mock 提问者问的是关键词问句，默认可别真值源
    （语义类别）解析不了，需要一个与 MockHost 同源的真值源。
    """
    def tf(p, question):
        for qa in p.golden_qa:
            if qa.get("contains") and qa["contains"] in question:
                return qa.get("answer")
        return None
    return tf


def _keyword_truth(puzzle):
    """question → golden_qa 子串匹配答案（tdhit 的 truth_fn 用，单参闭包）。"""
    tf = _keyword_truth_for(puzzle)
    return lambda q: tf(puzzle, q)


def train_questioner(train_set, host, episodes, seed, max_rounds):
    policy = TabularQPolicy(len(CATEGORY_NAMES), seed=seed)
    q = make_questioner(policy, explore=True)
    for ep in range(episodes):
        run_episode(train_set[ep % len(train_set)], host, q,
                    max_rounds=max_rounds, train=True)
    return policy


def train_host(host, train_set, questioner, episodes, max_rounds):
    """主持侧训练（提问者冻结）；host 传入即继续训练其 Q 表（Phase C 用）。"""
    for ep in range(episodes):
        run_episode(train_set[ep % len(train_set)], host, questioner,
                    max_rounds=max_rounds, train=False, train_host=True,
                    lie_cost=host.lie_cost)


def eval_clone(host):
    """评估态副本：纯贪心，共享 Q 表（评估不更新，安全；见 run_episode）。"""
    return DeceptiveHost(policy=host.policy, k=host.k,
                         lie_cost=host.lie_cost, explore=False, id=host.id)


def host_for(name, args, seed, trained):
    """每次评测构造新鲜主持（RandomLiarHost 的 RNG 按 seed 重置 → 单元格间配对）。"""
    if name == "truthful":
        return CategoryHost()
    if name == "random-liar":
        return RandomLiarHost(max_rounds=args.max_rounds, k=args.k, seed=seed)
    if name == "first-k-liar":
        return FirstKLiarHost(k=args.k)
    return eval_clone(trained)


# ---------------- 评测 ----------------
def evaluate_cell(puzzles, host, policy_q, repeats, max_rounds):
    """该单元格的逐题指标行（保留原始行——报告不只有均值）。"""
    q = make_questioner(policy_q, explore=False)
    rows = []
    for p in puzzles:
        tf = make_truth_fn(p)
        for _ in range(repeats):
            out = run_episode(p, host, q, max_rounds=max_rounds, train=False)
            tr = out["transcript"]
            qs = [m["content"] for m in tr if m.get("role") == "question"]
            n_lies, pending = 0, None
            for m in tr:
                if m.get("role") == "question":
                    pending = m.get("content")
                elif m.get("role") == "answer" and pending is not None:
                    t = tf(pending)
                    if t in ("是", "否") and (m.get("content") or "").strip() != t:
                        n_lies += 1
                    pending = None
            rows.append({
                "soup_id": p.soup_id,
                "tdhit": tdhit(tr, tf),
                "dhit": distinct_hit_rate(tr),
                "f1": out["f1"],
                "reward": out["reward"],
                "repeat_rate": 1.0 - len(set(qs)) / len(qs) if qs else 0.0,
                "n_lies": n_lies,
            })
    return rows


def cell_mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def per_lie_damage(base_value: float, host_value: float, n_lies: float):
    """每谎伤害 =（truthful 基线 − 该主持指标）/ 谎言数。

    主持恒真（n_lies<=0）时返回 **None**——处理未激活（§6-A 发现 2：
    λ 高于伤害收益时主持理性地不说谎）。刻意不返回 0.0：0.0 会被读成
    「有谎言但零伤害」，与「没有谎言」是两回事，混淆两者会让 E1' 在
    惰性区间假装被判定过。
    """
    if n_lies <= 0:
        return None
    return (base_value - host_value) / n_lies


# ---------------- 单 seed 全流程 ----------------
def run_seed(seed, puzzles, args, lam):
    train_set, heldout = split_puzzles(puzzles, args.train_size, seed)
    n_cat = len(CATEGORY_NAMES)

    # Q-honest：仅对诚实主持训练（= 当前 L2 版本口径，复跑保证同基准）
    qh = train_questioner(train_set, CategoryHost(), args.episodes_q,
                          seed, args.max_rounds)

    # Phase H：主持预训练 vs 冻结 DistinctRandomPolicy
    rnd_q = make_questioner(DistinctRandomPolicy(n_cat, seed=seed), explore=True)
    host = DeceptiveHost(k=args.k, lie_cost=lam,
                         epsilon=EPS_HOST, seed=seed)
    train_host(host, train_set, rnd_q, args.episodes_host, args.max_rounds)

    # Phase Q：对抗训练（主持冻结 ε=0.1 随机化）
    host.policy.epsilon = EPS_FROZEN
    host.explore = True
    qa = train_questioner(train_set, host, args.episodes_q, seed, args.max_rounds)

    # Phase C：共演 R 轮，记录轨迹（震荡诊断用）
    coev = []
    for _ in range(args.coevolve_rounds):
        frozen_q = make_questioner(qa, explore=False)
        train_host(host, train_set, frozen_q, args.episodes_host, args.max_rounds)
        host.policy.epsilon = EPS_FROZEN
        host.explore = True
        qa = train_questioner(train_set, host, args.episodes_q, seed, args.max_rounds)
        snap = evaluate_cell(heldout, eval_clone(host), qa,
                             args.repeats, args.max_rounds)
        coev.append(cell_mean(snap, "tdhit"))

    # 评测矩阵（held-out 主，训练集副）
    questioners = (("Q-honest", qh), ("Q-adv", qa))
    host_names = ("truthful", "random-liar", "first-k-liar", "trained-deceptive")
    matrix = {}
    for qn, qp in questioners:
        for hn in host_names:
            h = host_for(hn, args, seed, host)
            matrix[(qn, hn)] = {
                "heldout": evaluate_cell(heldout, h, qp, args.repeats,
                                         args.max_rounds),
                "train": evaluate_cell(train_set, h, qp, args.repeats,
                                       args.max_rounds),
            }

    # 预注册判据（§6 原文 + §6-A 修订）
    def m(qn, hn, split="heldout"):
        return cell_mean(matrix[(qn, hn)][split], "tdhit")

    def f1(qn, hn, split="heldout"):
        return cell_mean(matrix[(qn, hn)][split], "f1")

    def dh(qn, hn, split="heldout"):
        return cell_mean(matrix[(qn, hn)][split], "dhit")

    def lies(qn, hn):
        return cell_mean(matrix[(qn, hn)]["heldout"], "n_lies")

    def damage(metric_fn, hn):
        """每谎伤害（语义与 None 约定见模块级 per_lie_damage）。"""
        return per_lie_damage(metric_fn("Q-honest", "truthful"),
                              metric_fn("Q-honest", hn),
                              lies("Q-honest", hn))

    verdict = {
        # E1' 主判据：F1 每谎伤害（§6-A 发现 1：tdhit 版是恒等式，已弃用）
        "e1p_f1_trained": damage(f1, "trained-deceptive"),
        "e1p_f1_random": damage(f1, "random-liar"),
        # E1'' 辅证：dhit 每谎伤害
        "e1pp_dhit_trained": damage(dh, "trained-deceptive"),
        "e1pp_dhit_random": damage(dh, "random-liar"),
        # 恒等式存档：tdhit 每谎伤害（理论值恒 1/max_rounds）
        "arch_tdhit_trained": damage(m, "trained-deceptive"),
        "arch_tdhit_random": damage(m, "random-liar"),
        # E2 主判据（tdhit，预期可能趋平）+ E2-f1 次要读数（F1）
        "e2_tdhit_gain": m("Q-adv", "trained-deceptive") - m("Q-honest", "trained-deceptive"),
        "e2_f1_gain": f1("Q-adv", "trained-deceptive") - f1("Q-honest", "trained-deceptive"),
        "cost_truthful_tdhit": m("Q-honest", "truthful") - m("Q-adv", "truthful"),
        "cost_truthful_f1": f1("Q-honest", "truthful") - f1("Q-adv", "truthful"),
        "n_lies_trained": lies("Q-honest", "trained-deceptive"),
    }
    d_tr, d_rn = verdict["e1p_f1_trained"], verdict["e1p_f1_random"]
    verdict["e1p_pass"] = (
        None if (d_tr is None or d_rn is None) else d_tr > d_rn)
    verdict["e2_pass"] = verdict["e2_tdhit_gain"] > 0
    return {"seed": seed, "lam": lam, "matrix": matrix, "verdict": verdict,
            "coev": coev, "n_train": len(train_set), "n_held": len(heldout),
            "host_states": len(host.policy.Q)}


# ---------------- E3：混合对局贯通 ----------------
def run_e3(puzzle, args, lam):
    """混合对局端到端（真实引擎 TurtleSoupGame + MockJudge）。"""
    rows = []
    # (a) RL 提问者（语义问句）× MARL 主持（类别真值）
    host = DeceptiveHost(k=args.k, lie_cost=lam,
                         epsilon=EPS_HOST, seed=args.e3_seed)
    host.explore = False
    q = make_questioner(TabularQPolicy(len(CATEGORY_NAMES), seed=args.e3_seed),
                        explore=False)
    game = TurtleSoupGame(puzzle, host, [q], judge=MockJudge(),
                          max_rounds=args.max_rounds)
    res = game.run()
    d = res.to_dict()
    row = metrics_from_result(d)
    row["tdhit"] = tdhit(d["transcript"], make_truth_fn(puzzle))
    row["mix"] = "RL-questioner × MARL-host"
    rows.append(row)
    # (b) mock 提问者（关键词问句）× MARL 主持（golden_qa 真值注入）
    host2 = DeceptiveHost(k=args.k, lie_cost=lam,
                          epsilon=EPS_HOST, seed=args.e3_seed,
                          truth_fn=_keyword_truth_for(puzzle))
    host2.explore = False
    game2 = TurtleSoupGame(puzzle, host2, [MockQuestioner(id="mock", seed=1)],
                           judge=MockJudge(), max_rounds=args.max_rounds)
    res2 = game2.run()
    d2 = res2.to_dict()
    row2 = metrics_from_result(d2)
    row2["tdhit"] = tdhit(d2["transcript"], _keyword_truth(puzzle))
    row2["mix"] = "mock-questioner × MARL-host"
    rows.append(row2)
    return rows


# ---------------- 报告 ----------------
def _fmt(x):
    return f"{x:+.3f}" if isinstance(x, float) else str(x)


def _fmt_or_na(x, spec=".4f"):
    return "N/A（恒真）" if x is None else format(x, spec)


def write_report(path, args, runs, e3_rows, elapsed):
    """runs = [(lam, [per_seed...]), ...] 按 λ 分组；per_seed 见 run_seed。"""
    q_names = ("Q-honest", "Q-adv")
    h_names = ("truthful", "random-liar", "first-k-liar", "trained-deceptive")
    lines = []
    lines.append("# P2-B B-1 · MARL 主持对抗自对弈报告（λ 扫描 · 修订后判据）\n")
    lines.append(f"- 配置：seeds={args.seeds} | k={args.k} | λ 扫描="
                 f"{[r[0] for r in runs]} | rounds={args.max_rounds} | "
                 f"episodes H/Q={args.episodes_host}/{args.episodes_q} | "
                 f"共演 {args.coevolve_rounds} 轮 | eval repeats={args.repeats}")
    lines.append(f"- 协议：Phase H（主持 vs 冻结 DistinctRandom）→ Phase Q"
                 f"（提问者 vs 冻结主持 ε={EPS_FROZEN}）→ Phase C 共演；"
                 f"评估一律纯贪心")
    lines.append("- 判据：**预注册 §6 + 预实验修订 §6-A**（修订发生在确认性"
                 "实验之前，方向由恒等式证明决定）：E1' 每谎 **F1** 伤害 > "
                 "random-liar（主）；E1'' dhit 为辅证；E2 tdhit 鲁棒增益"
                 "（预期可能趋平）；E2-f1 F1 鲁棒增益（次要）；E3 贯通。"
                 "**不为凑正结果改判据**\n")
    lines.append(f"- ⚠️ tdhit 每谎伤害在**提问者不复问时**是恒等式 "
                 f"1/max_rounds（§6-A 发现 1；复问区间被首答去重吸收，"
                 f"damage ≤ 1/R）。任何不复问区间的说谎主持恒等于 "
                 f"{1.0 / args.max_rounds:.3f}。原 E1 判据因此弃用，"
                 f"本报告仅存档对照\n")

    for lam, per_seed in runs:
        n_train = per_seed[0]["n_train"]
        n_held = per_seed[0]["n_held"]
        lines.append(f"\n## λ = {lam}\n")
        lines.append(f"- 训练/测试题 {n_train}/{n_held} | "
                     f"主持平均谎言数（trained-deceptive）"
                     f"{cell_mean([r for ps in per_seed for r in ps['matrix'][('Q-honest', 'trained-deceptive')]['heldout']], 'n_lies'):.2f}"
                     f" / 配额 {args.k}\n")

        lines.append("### 逐种子矩阵（held-out 题目，tdhit / dhit / F1 / 复问率 / n_lies）\n")
        lines.append("| seed | 提问者 | 主持 | tdhit | dhit | F1 | 复问率 | n_lies |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for ps in per_seed:
            for qn in q_names:
                for hn in h_names:
                    rows = ps["matrix"][(qn, hn)]["heldout"]
                    lines.append(
                        f"| {ps['seed']} | {qn} | {hn} | "
                        f"{cell_mean(rows, 'tdhit'):.3f} | {cell_mean(rows, 'dhit'):.3f} | "
                        f"{cell_mean(rows, 'f1'):.3f} | {cell_mean(rows, 'repeat_rate'):.3f} | "
                        f"{cell_mean(rows, 'n_lies'):.2f} |")

        lines.append("\n### 预注册判据逐种子判定\n")
        lines.append("| seed | E1' F1伤害 trained | E1' random | E1' | "
                     "E1'' dhit伤害 trained | random | E2 tdhit增益 | E2 | "
                     "E2-f1 F1增益 | truthful代价(tdhit) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ps in per_seed:
            v = ps["verdict"]
            lines.append(
                f"| {ps['seed']} | {_fmt_or_na(v['e1p_f1_trained'])} | "
                f"{_fmt_or_na(v['e1p_f1_random'])} | "
                f"{'N/A' if v['e1p_pass'] is None else ('✅' if v['e1p_pass'] else '❌')} | "
                f"{_fmt_or_na(v['e1pp_dhit_trained'])} | "
                f"{_fmt_or_na(v['e1pp_dhit_random'])} | "
                f"{v['e2_tdhit_gain']:+.3f} | "
                f"{'✅' if v['e2_pass'] else '❌'} | "
                f"{v['e2_f1_gain']:+.3f} | "
                f"{v['cost_truthful_tdhit']:+.3f} |")
        n = len(per_seed)
        judged = [ps for ps in per_seed if ps["verdict"]["e1p_pass"] is not None]
        e1_yes = sum(1 for ps in judged if ps["verdict"]["e1p_pass"])
        e2_yes = sum(1 for ps in per_seed if ps["verdict"]["e2_pass"])
        lines.append(f"\n**λ={lam} 多数裁决：E1' {e1_yes}/{len(judged)} 通过"
                     f"（{len(judged)}/{n} 个 seed 主持实际说谎，其余恒真=处理未激活）；"
                     f"E2 {e2_yes}/{n} 通过**（判据：跨 seed 多数为正）\n")
        # 恒等式存档
        arch_t = per_seed[0]["verdict"]["arch_tdhit_trained"]
        arch_r = per_seed[0]["verdict"]["arch_tdhit_random"]
        lines.append(f"- 恒等式存档（tdhit 每谎伤害）：trained "
                     f"{_fmt_or_na(arch_t)} / random {_fmt_or_na(arch_r)} / "
                     f"理论值 {1.0 / args.max_rounds:.3f}\n")
        if any(ps["coev"] for ps in per_seed):
            lines.append("### 共演轨迹（Phase C 逐轮 tdhit，震荡诊断）\n")
            for ps in per_seed:
                if ps["coev"]:
                    lines.append(f"- seed={ps['seed']}: "
                                 + " → ".join(f"{x:.3f}" for x in ps["coev"]))
            lines.append("")

    # λ 区间图（处理激活性）
    lines.append("\n## λ 区间图（§6-A 发现 2：默认 λ=0.05 主持恒真）\n")
    lines.append("| λ | 平均谎言数 trained | E1' F1伤害 trained | E1' random | "
                 "E1' 多数裁决 | E2 多数裁决 |")
    lines.append("|---|---|---|---|---|---|")
    for lam, per_seed in runs:
        rows = [r for ps in per_seed
                for r in ps["matrix"][("Q-honest", "trained-deceptive")]["heldout"]]
        judged = [ps for ps in per_seed if ps["verdict"]["e1p_pass"] is not None]
        e1_yes = sum(1 for ps in judged if ps["verdict"]["e1p_pass"])
        e2_yes = sum(1 for ps in per_seed if ps["verdict"]["e2_pass"])
        d_tr = [ps["verdict"]["e1p_f1_trained"] for ps in judged
                if ps["verdict"]["e1p_f1_trained"] is not None]
        d_rn = [ps["verdict"]["e1p_f1_random"] for ps in judged
                if ps["verdict"]["e1p_f1_random"] is not None]
        lines.append(
            f"| {lam} | {cell_mean(rows, 'n_lies'):.2f} | "
            f"{(sum(d_tr) / len(d_tr)) if d_tr else 'N/A'} | "
            f"{(sum(d_rn) / len(d_rn)) if d_rn else 'N/A'} | "
            f"{e1_yes}/{len(judged)} | {e2_yes}/{len(per_seed)} |")
    lines.append("")

    lines.append("\n## E3 混合对局（真实引擎，跑通即过）\n")
    lines.append("| 组合 | vra | yn | dhit | tdhit | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for r in e3_rows:
        lines.append(f"| {r['mix']} | {r['vra']:.3f} | {r['yn_rate']:.3f} | "
                     f"{r['dhit']:.3f} | {r['tdhit']:.3f} | {r['f1']:.3f} |")
    lines.append("")

    lines.append("\n## 失败模式对照（§7 pre-declared 五条 + §6-A 发现 2 第六条）\n")
    lines.append("| 模式 | 诊断 | 本 run 观察 |")
    lines.append("|---|---|---|")
    rows = [r for lam, per_seed in runs for ps in per_seed
            for r in ps["matrix"][("Q-honest", "trained-deceptive")]["heldout"]]
    lines.append(f"| ① 主持退化为「有配额就花」 | 谎言位置分布 | 平均谎言数 "
                 f"{cell_mean(rows, 'n_lies'):.2f} / 配额 {args.k}（跨 λ） |")
    lines.append(f"| ② 提问者复问刷哨 | 复问率↑ 而 tdhit 不涨 | trained-deceptive 下 "
                 f"复问率 {cell_mean(rows, 'repeat_rate'):.3f} |")
    lines.append("| ③ 共演震荡 | Phase C 轨迹循环追逐 | 见各 λ 共演轨迹 |")
    lines.append("| ④ 12 题过拟合 | held-out vs 训练集结论反向 | 双方状态均题目无关（§3.2）"
                 "，held-out 为准 |")
    lines.append("| ⑤ 信号弱 | Q-adv 与 Q-honest 无差别 | E2/E2-f1 判定即回答 |")
    lines.append("| ⑥（§6-A 新观察）主持恒真 | λ 高于伤害收益 → 0 谎 | 见 λ 区间图"
                 "谎言数列 |\n")

    lines.append("\n## 训练集对照（held-out 结论的旁证，按 λ 汇总）\n")
    lines.append("| λ | seed | 提问者 | 主持 | tdhit(train) |")
    lines.append("|---|---|---|---|---|")
    for lam, per_seed in runs:
        for ps in per_seed:
            for qn in q_names:
                for hn in h_names:
                    rows = ps["matrix"][(qn, hn)]["train"]
                    lines.append(f"| {lam} | {ps['seed']} | {qn} | {hn} | "
                                 f"{cell_mean(rows, 'tdhit'):.3f} |")
    lines.append(f"\n---\n总耗时 {elapsed:.1f}s。诚实边界：双方均为表格 Q（无函数\n"
                 f"逼近），本阶段目标是「评测台能评测 MARL agent」，不是提出新\n"
                 f"MARL 算法；E1'/E2 全负时阶段目标仍由 E3 判定达成与否。\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDA_SWEEP),
                    help="λ 扫描区间（§6-A 发现 2）")
    ap.add_argument("--episodes-host", type=int, default=2000)
    ap.add_argument("--episodes-q", type=int, default=4000)
    ap.add_argument("--k", type=int, default=K_DEFAULT, help="每局谎言配额")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--train-size", type=int, default=8,
                    help="训练题数（12 题题库 → held-out = 12 − 此值）")
    ap.add_argument("--coevolve-rounds", type=int, default=2,
                    help="Phase C 共演轮数（0 = 只交付 Phase H+Q）")
    ap.add_argument("--repeats", type=int, default=10, help="评测时每题重复局数")
    ap.add_argument("--e3-seed", type=int, default=0)
    ap.add_argument("--e3-lie-cost", type=float, default=0.005,
                    help="E3 混合对局的 λ（基础设施贯通，非判据）")
    ap.add_argument("--out", default=os.path.join("logs", "marl_report.md"))
    args = ap.parse_args()

    puzzles = load_puzzles()
    t0 = time.time()
    print(f"[marl] 题库 {len(puzzles)} 题 | seeds={args.seeds} | k={args.k} "
          f"| λ 扫描={args.lambdas} | H/Q episodes={args.episodes_host}/"
          f"{args.episodes_q} | 共演 {args.coevolve_rounds} 轮")

    runs = []
    for lam in args.lambdas:
        per_seed = []
        for seed in args.seeds:
            ts = time.time()
            ps = run_seed(seed, puzzles, args, lam)
            per_seed.append(ps)
            v = ps["verdict"]
            print(f"[λ={lam} seed {seed}] {time.time() - ts:.1f}s | "
                  f"谎言数 {v['n_lies_trained']:.2f} | "
                  f"E1' F1伤害 trained {_fmt_or_na(v['e1p_f1_trained'])} vs "
                  f"random {_fmt_or_na(v['e1p_f1_random'])} → "
                  f"{'N/A' if v['e1p_pass'] is None else ('PASS' if v['e1p_pass'] else 'FAIL')} | "
                  f"E2 tdhit增益 {v['e2_tdhit_gain']:+.3f} → "
                  f"{'PASS' if v['e2_pass'] else 'FAIL'} | "
                  f"E2-f1 {v['e2_f1_gain']:+.3f} | "
                  f"truthful代价 {v['cost_truthful_tdhit']:+.3f} | "
                  f"主持状态数 {ps['host_states']}")
        runs.append((lam, per_seed))

    e3_rows = run_e3(puzzles[0], args, args.e3_lie_cost)
    print(f"[e3] 混合对局 {len(e3_rows)} 组跑通："
          + "；".join(f"{r['mix']} tdhit={r['tdhit']:.3f}" for r in e3_rows))

    write_report(args.out, args, runs, e3_rows, time.time() - t0)
    print(f"[done] 报告已写入 {args.out} | 总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
