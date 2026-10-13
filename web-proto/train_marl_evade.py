#!/usr/bin/env python3
"""P2-B B-1b 训练与评测入口：主持搪塞轴 · 双配额零和自对弈。

设计见 DESIGN-P2B1B-EVASION.md（前置阅读：DESIGN-P2B-MARL.md——B-1
已给出照实负结果，本阶段检验搪塞轴是否带来 B-1 缺失的可学结构）。

协议（§4 交替冻结自对弈，同 B-1）：
  Phase H  主持预训练：提问者冻结为 DistinctRandomPolicy（不复问⇒不转化），
           训练 EvasiveDeceptiveHost（三动作 + 双配额）
  Phase Q  对抗训练：主持冻结（ε=0.1 随机化），训练提问者（可学转化）
  Phase C  共演 R 轮：交替 [主持 vs 冻结提问者(评估态)] →
           [提问者 vs 冻结主持(ε=0.1)]，输出共演轨迹诊断

评测（§5 矩阵，held-out 题目为主、训练集副）：
  提问者 {Q-honest（仅对诚实主持训练）, Q-adv（对抗训练产物）}
  × 主持 {truthful, random-evader（E1b 关键对照）, random-liar（谎言轴
          参照）, trained-evasive（本阶段处理）}
  指标：tdhit / dhit / F1 / reward / 复问率 / 谎言数 / 搪塞数 /
        转化率（被搪塞问句后被复问的比例）/ 已答问句复问率

判据（§6 预注册，先定后跑，不为凑正结果改判据）：
  E1b   训练主持每**搪塞** F1 伤害 > random-evader（同 k_e/ε/λ）——主判据；
        主持未搪塞（n_evades=0）时判 N/A（处理未激活），不判 0/0
  E1b'  每搪塞 tdhit 伤害存档（无转化区间恒等式 1/R；转化区间 2/(R+1)，
        见 §6-B——tdhit 结构性惩罚复问，读它必须连带读转化率）
  E2b   转化率：Q-adv 在 trained-evasive 下对「被搪塞问句」的复问率 >
        Q-honest 同环境（严格大于）；同时报「已回答问句复问率」以区分
        「学了转化」与「普遍更爱复问」
  E2b'  Q-adv 在 trained-evasive 下 F1 > Q-honest（鲁棒读数，照实报）
  E3b   混合对局（mock/RL 提问 × evasive 主持）端到端跑通

ε 扫描 {0, 0.01, 0.05, 0.2, 0.6, 1.3}：区间理由见设计文档 §3.3——
搪塞每步收益 = 1.2 − ε（提问者无信息惩罚只有 0.2，与谎言轴的 λ 尺度
不可比），ε≈0.2 是「经得起全额转化」的边界，ε≈1.2 才压制每步意义。
λ 固定 0.01（B-1 活跃区间；λ×ε 交互是后续，本文不做）。

执行纪律（项目 2026-10-13 起）：训练/评测一律上服务器 172.16.222.99
（纯 Python 零依赖，python3 直跑）。本机只跑轻量单测与 sanity。
"""
import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.game import TurtleSoupGame
from engine.puzzles import load_puzzles
from engine.evaluate import distinct_hit_rate, metrics_from_result, tdhit
from engine.semantic import (
    CATEGORY_NAMES, CategoryHost, category_answer, parse_category,
)
from eval_generalization import DistinctRandomPolicy, split_puzzles
from players.host_rl import (
    ACTION_EVADE, ACTION_LIE, ACTION_TRUTH, EVASIVE_ANSWER,
    DeceptiveHost, EvasiveDeceptiveHost, FirstKLiarHost, RandomEvaderHost,
    RandomLiarHost,
)
from players.mock import MockJudge, MockQuestioner
from players.rl import (
    EVASIVE_ANSWERS, RLQuestioner, TabularQPolicy, run_episode,
)

EPS_HOST = 0.15        # 主持训练探索率
EPS_FROZEN = 0.1       # Phase Q/C 冻结主持的随机化
K_DEFAULT = 2          # 每局谎言配额
K_E_DEFAULT = 2        # 每局搪塞配额（B-1b）
LAMBDA_FIXED = 0.01    # λ 固定在 B-1 活跃区间（§1：不做双扫描）
EPSILON_SWEEP = (0.0, 0.01, 0.05, 0.2, 0.6, 1.3)  # §3.3 区间理由


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
    """(puzzle, question) → golden_qa 子串匹配答案（E3b 注入用）。"""
    def tf(p, question):
        for qa in p.golden_qa:
            if qa.get("contains") and qa["contains"] in question:
                return qa.get("answer")
        return None
    return tf


def _keyword_truth(puzzle):
    tf = _keyword_truth_for(puzzle)
    return lambda q: tf(puzzle, q)


def train_questioner(train_set, host, episodes, seed, max_rounds):
    policy = TabularQPolicy(len(CATEGORY_NAMES), seed=seed)
    q = make_questioner(policy, explore=True)
    for ep in range(episodes):
        run_episode(train_set[ep % len(train_set)], host, q,
                    max_rounds=max_rounds, train=True)
    return policy


def train_host(host, train_set, questioner, episodes, max_rounds,
               lie_cost, evade_cost):
    """主持侧训练（提问者冻结）；host 传入即继续训练其 Q 表（Phase C 用）。"""
    for ep in range(episodes):
        run_episode(train_set[ep % len(train_set)], host, questioner,
                    max_rounds=max_rounds, train=False, train_host=True,
                    lie_cost=lie_cost, evade_cost=evade_cost)


def eval_clone(host):
    """评估态副本：纯贪心，共享 Q 表（评估不更新，安全）。"""
    if isinstance(host, EvasiveDeceptiveHost):
        return EvasiveDeceptiveHost(policy=host.policy, k=host.k, k_e=host.k_e,
                                    lie_cost=host.lie_cost,
                                    evade_cost=host.evade_cost,
                                    explore=False, id=host.id)
    return DeceptiveHost(policy=host.policy, k=host.k,
                         lie_cost=host.lie_cost, explore=False, id=host.id)


def host_for(name, args, seed, trained):
    """每次评测构造新鲜主持（RandomLiar/Evader 的 RNG 按 seed 重置 → 配对）。"""
    if name == "truthful":
        return CategoryHost()
    if name == "random-liar":
        return RandomLiarHost(max_rounds=args.max_rounds, k=args.k, seed=seed)
    if name == "random-evader":
        return RandomEvaderHost(max_rounds=args.max_rounds, k_e=args.k_e,
                                seed=seed)
    return eval_clone(trained)


# ---------------- 指标助手 ----------------
def _pairs(transcript):
    out, pending = [], None
    for m in transcript:
        if m.get("role") == "question":
            pending = m.get("content", "")
        elif m.get("role") == "answer" and pending is not None:
            out.append((pending, m.get("content", "")))
            pending = None
    return out


def conversion_rate(pairs):
    """被搪塞过的去重问句中，搪塞之后再次得到回答（转化成功）的比例。

    分母 = 被搪塞的去重问句数；无搪塞返回 None（无定义——刻意不返回
    0.0：0.0 会被读成「有搪塞但零转化」，与「没有搪塞」是两回事）。
    """
    evaded, converted = set(), set()
    for q, a in pairs:
        if a in EVASIVE_ANSWERS:
            evaded.add(q)
        elif q in evaded:
            converted.add(q)
    if not evaded:
        return None
    return len(converted) / len(evaded)


def reask_answered_rate(pairs):
    """得到过是/否 回答的去重问句中，之后又被问一次（真复问）的比例。

    E2b 的区分读数：转化率涨 ≠ 学了转化（可能只是普遍更爱复问）。
    """
    answered, reasked = set(), set()
    for q, a in pairs:
        if a in ("是", "否"):
            if q in answered:
                reasked.add(q)
            answered.add(q)
    if not answered:
        return None
    return len(reasked) / len(answered)


def count_evades(pairs):
    return sum(1 for _, a in pairs if a in EVASIVE_ANSWERS)


# ---------------- 评测 ----------------
def evaluate_cell(puzzles, host, policy_q, repeats, max_rounds):
    """该单元格的逐题指标行（保留原始行——报告不只有均值）。"""
    q = make_questioner(policy_q, explore=False)
    rows = []
    for p in puzzles:
        tf = make_truth_fn(p)
        for _ in range(repeats):
            out = run_episode(p, host, q, max_rounds=max_rounds, train=False)
            pairs = _pairs(out["transcript"])
            qs = [q_ for q_, _ in pairs]
            n_lies, n_evades = 0, 0
            for question, answer in pairs:
                t = tf(question)
                if t in ("是", "否"):
                    if answer in EVASIVE_ANSWERS:
                        n_evades += 1
                    elif answer.strip() != t:
                        n_lies += 1
            rows.append({
                "soup_id": p.soup_id,
                "tdhit": tdhit(out["transcript"], tf),
                "dhit": distinct_hit_rate(out["transcript"]),
                "f1": out["f1"],
                "reward": out["reward"],
                "repeat_rate": 1.0 - len(set(qs)) / len(qs) if qs else 0.0,
                "n_lies": n_lies,
                "n_evades": n_evades,
                "conversion_rate": conversion_rate(pairs),
                "reask_answered_rate": reask_answered_rate(pairs),
            })
    return rows


def cell_mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def cell_mean_opt(rows, key):
    """可缺指标（None = 无定义）的均值；全 None 返回 None（不伪装成 0）。"""
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def per_evade_damage(base_value: float, host_value: float, n_evades: float):
    """每搪塞伤害 =（truthful 基线 − 该主持指标）/ 搪塞数。

    主持未搪塞（n_evades<=0）时返回 None——处理未激活。刻意不返回 0.0：
    0.0 会被读成「有搪塞但零伤害」，与「没有搪塞」是两回事（B-1 的
    per_lie_damage 同款纪律）。
    """
    if n_evades <= 0:
        return None
    return (base_value - host_value) / n_evades


# ---------------- 单 seed 全流程 ----------------
def run_seed(seed, puzzles, args, eps):
    train_set, heldout = split_puzzles(puzzles, args.train_size, seed)
    n_cat = len(CATEGORY_NAMES)

    # Q-honest：仅对诚实主持训练（= 当前 L2 版本口径）
    qh = train_questioner(train_set, CategoryHost(), args.episodes_q,
                          seed, args.max_rounds)

    # Phase H：主持预训练 vs 冻结 DistinctRandomPolicy（不复问⇒不转化）
    rnd_q = make_questioner(DistinctRandomPolicy(n_cat, seed=seed), explore=True)
    host = EvasiveDeceptiveHost(k=args.k, k_e=args.k_e,
                                lie_cost=LAMBDA_FIXED, evade_cost=eps,
                                epsilon=EPS_HOST, seed=seed)
    train_host(host, train_set, rnd_q, args.episodes_host, args.max_rounds,
               LAMBDA_FIXED, eps)

    # Phase Q：对抗训练（主持冻结 ε=0.1 随机化）——提问者在此面对搪塞
    host.policy.epsilon = EPS_FROZEN
    host.explore = True
    qa = train_questioner(train_set, host, args.episodes_q, seed,
                          args.max_rounds)

    # Phase C：共演 R 轮，记录轨迹（震荡诊断用）
    coev = []
    for _ in range(args.coevolve_rounds):
        frozen_q = make_questioner(qa, explore=False)
        train_host(host, train_set, frozen_q, args.episodes_host,
                   args.max_rounds, LAMBDA_FIXED, eps)
        host.policy.epsilon = EPS_FROZEN
        host.explore = True
        qa = train_questioner(train_set, host, args.episodes_q, seed,
                              args.max_rounds)
        snap = evaluate_cell(heldout, eval_clone(host), qa,
                             args.repeats, args.max_rounds)
        coev.append(cell_mean(snap, "tdhit"))

    # 评测矩阵（held-out 主，训练集副）
    questioners = (("Q-honest", qh), ("Q-adv", qa))
    host_names = ("truthful", "random-evader", "random-liar",
                  "trained-evasive")
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

    # 预注册判据（§6）
    def m(qn, hn, split="heldout"):
        return cell_mean(matrix[(qn, hn)][split], "tdhit")

    def f1(qn, hn, split="heldout"):
        return cell_mean(matrix[(qn, hn)][split], "f1")

    def dh(qn, hn, split="heldout"):
        return cell_mean(matrix[(qn, hn)][split], "dhit")

    def evades(qn, hn):
        return cell_mean(matrix[(qn, hn)]["heldout"], "n_evades")

    def lies(qn, hn):
        return cell_mean(matrix[(qn, hn)]["heldout"], "n_lies")

    def damage(metric_fn, hn):
        """每搪塞伤害（None 约定见模块级 per_evade_damage）。"""
        return per_evade_damage(metric_fn("Q-honest", "truthful"),
                                metric_fn("Q-honest", hn),
                                evades("Q-honest", hn))

    def conv(qn, hn):
        return cell_mean_opt(matrix[(qn, hn)]["heldout"], "conversion_rate")

    verdict = {
        # E1b 主判据：F1 每搪塞伤害
        "e1b_f1_trained": damage(f1, "trained-evasive"),
        "e1b_f1_random": damage(f1, "random-evader"),
        # E1b' 辅证：tdhit 每搪塞伤害（恒等式存档 + 转化区间口径 §6-B）
        "e1bp_tdhit_trained": damage(m, "trained-evasive"),
        "e1bp_tdhit_random": damage(m, "random-evader"),
        # 谎言轴参照（每谎 F1 伤害，B-1 同口径）
        "ref_f1_lie_trained": per_lie_damage(f1("Q-honest", "truthful"),
                                             f1("Q-honest", "trained-evasive"),
                                             lies("Q-honest", "trained-evasive")),
        "ref_f1_lie_random": per_lie_damage(f1("Q-honest", "truthful"),
                                            f1("Q-honest", "random-liar"),
                                            lies("Q-honest", "random-liar")),
        # E2b：转化率（被搪塞问句的复问率）Q-adv vs Q-honest
        "e2b_conv_qadv": conv("Q-adv", "trained-evasive"),
        "e2b_conv_qhonest": conv("Q-honest", "trained-evasive"),
        # E2b 区分读数：已回答问句的复问率（防「普遍更爱复问」冒充「学了转化」）
        "e2b_reask_ans_qadv": cell_mean_opt(
            matrix[("Q-adv", "trained-evasive")]["heldout"], "reask_answered_rate"),
        "e2b_reask_ans_qhonest": cell_mean_opt(
            matrix[("Q-honest", "trained-evasive")]["heldout"], "reask_answered_rate"),
        # E2b'：F1 鲁棒读数
        "e2bp_f1_gain": f1("Q-adv", "trained-evasive") - f1("Q-honest", "trained-evasive"),
        "cost_truthful_tdhit": m("Q-honest", "truthful") - m("Q-adv", "truthful"),
        "n_evades_trained": evades("Q-honest", "trained-evasive"),
        "n_evades_random": evades("Q-honest", "random-evader"),
        "n_lies_trained": lies("Q-honest", "trained-evasive"),
    }
    d_tr, d_rn = verdict["e1b_f1_trained"], verdict["e1b_f1_random"]
    verdict["e1b_pass"] = (
        None if (d_tr is None or d_rn is None) else d_tr > d_rn)
    c_qa, c_qh = verdict["e2b_conv_qadv"], verdict["e2b_conv_qhonest"]
    verdict["e2b_pass"] = (
        None if (c_qa is None or c_qh is None) else c_qa > c_qh)
    return {"seed": seed, "eps": eps, "matrix": matrix, "verdict": verdict,
            "coev": coev, "n_train": len(train_set), "n_held": len(heldout),
            "host_states": len(host.policy.Q)}


def per_lie_damage(base_value: float, host_value: float, n_lies: float):
    """每谎伤害（谎言轴参照列用；None 约定同 per_evade_damage）。"""
    if n_lies <= 0:
        return None
    return (base_value - host_value) / n_lies


# ---------------- E3b：混合对局贯通 ----------------
def run_e3(puzzle, args, eps):
    """混合对局端到端（真实引擎 TurtleSoupGame + MockJudge）。

    E3b 是基础设施判据（跑通即过）。主持用 explore=True（ε=0.15）以便
    观察搪塞/谎言行为——未经训练的 Q 表在 explore=False 下贪心退化到
    truth，混合对局将什么都不会发生（B-1 的 E3 即此形态，此处显式选择
    可观察形态并如实注明）。
    """
    rows = []
    # (a) RL 提问者（语义问句）× evasive 主持（类别真值）
    host = EvasiveDeceptiveHost(k=args.k, k_e=args.k_e,
                                lie_cost=LAMBDA_FIXED, evade_cost=eps,
                                epsilon=EPS_HOST, seed=args.e3_seed,
                                explore=True)
    q = make_questioner(TabularQPolicy(len(CATEGORY_NAMES), seed=args.e3_seed),
                        explore=False)
    game = TurtleSoupGame(puzzle, host, [q], judge=MockJudge(),
                          max_rounds=args.max_rounds)
    d = game.run().to_dict()
    row = metrics_from_result(d)
    row["tdhit"] = tdhit(d["transcript"], make_truth_fn(puzzle))
    pairs = _pairs(d["transcript"])
    row["n_evades"] = count_evades(pairs)
    row["mix"] = "RL-questioner × evasive-host"
    rows.append(row)
    # (b) mock 提问者（关键词问句）× evasive 主持（golden_qa 真值注入）
    host2 = EvasiveDeceptiveHost(k=args.k, k_e=args.k_e,
                                 lie_cost=LAMBDA_FIXED, evade_cost=eps,
                                 epsilon=EPS_HOST, seed=args.e3_seed,
                                 truth_fn=_keyword_truth_for(puzzle),
                                 explore=True)
    game2 = TurtleSoupGame(puzzle, host2, [MockQuestioner(id="mock", seed=1)],
                           judge=MockJudge(), max_rounds=args.max_rounds)
    d2 = game2.run().to_dict()
    row2 = metrics_from_result(d2)
    row2["tdhit"] = tdhit(d2["transcript"], _keyword_truth(puzzle))
    row2["n_evades"] = count_evades(_pairs(d2["transcript"]))
    row2["mix"] = "mock-questioner × evasive-host"
    rows.append(row2)
    return rows


# ---------------- 报告 ----------------
def _fmt(x):
    return f"{x:+.3f}" if isinstance(x, float) else str(x)


def _fmt_or_na(x, spec=".4f"):
    return "N/A（未激活）" if x is None else format(x, spec)


def _fmt_or_na_conv(x):
    return "N/A（无搪塞）" if x is None else format(x, ".3f")


def write_report(path, args, runs, e3_rows, elapsed):
    """runs = [(eps, [per_seed...]), ...] 按 ε 分组；per_seed 见 run_seed。"""
    q_names = ("Q-honest", "Q-adv")
    h_names = ("truthful", "random-evader", "random-liar", "trained-evasive")
    lines = []
    lines.append("# P2-B B-1b · 主持搪塞轴对抗自对弈报告（ε 扫描 · 预注册判据）\n")
    lines.append(f"- 配置：seeds={args.seeds} | k={args.k} | k_e={args.k_e} | "
                 f"λ={LAMBDA_FIXED}（固定，B-1 活跃区间） | ε 扫描="
                 f"{[r[0] for r in runs]} | rounds={args.max_rounds} | "
                 f"episodes H/Q={args.episodes_host}/{args.episodes_q} | "
                 f"共演 {args.coevolve_rounds} 轮 | eval repeats={args.repeats}")
    lines.append("- 协议：Phase H（主持 vs 冻结 DistinctRandom，不复问⇒不转化）"
                 "→ Phase Q（提问者 vs 冻结主持 ε=0.1，首次面对搪塞）→ "
                 "Phase C 共演；评估一律纯贪心")
    lines.append("- 判据（预注册，先定后跑）：E1b 每搪塞 **F1** 伤害 > "
                 "random-evader（主，未搪塞判 N/A）；E1b' tdhit 存档"
                 "（无转化区间 ≡1/R，转化区间 2/(R+1)，§6-B）；E2b 转化率 "
                 "Q-adv > Q-honest（严格）；E2b' F1 鲁棒读数；E3b 贯通。"
                 "**不为凑正结果改判据**\n")
    lines.append("- ⚠️ 搪塞每步收益 = 1.2 − ε（提问者无信息惩罚仅 0.2，"
                 "与谎言轴 λ 尺度不可比）：ε<0.2 时即使被全额转化，搪塞"
                 "两步合计仍优于直接回答；ε≈1.2 才压制每步意义。扫描区间"
                 "即按此四档设计（§3.3）\n")

    for eps, per_seed in runs:
        n_train = per_seed[0]["n_train"]
        n_held = per_seed[0]["n_held"]
        lines.append(f"\n## ε = {eps}\n")
        lines.append(f"- 训练/测试题 {n_train}/{n_held} | "
                     f"trained-evasive 平均搪塞数 "
                     f"{cell_mean([r for ps in per_seed for r in ps['matrix'][('Q-honest', 'trained-evasive')]['heldout']], 'n_evades'):.2f}"
                     f" / 配额 {args.k_e} | 平均谎言数 "
                     f"{cell_mean([r for ps in per_seed for r in ps['matrix'][('Q-honest', 'trained-evasive')]['heldout']], 'n_lies'):.2f}"
                     f" / 配额 {args.k}\n")

        lines.append("### 逐种子矩阵（held-out 题目，tdhit / dhit / F1 / 复问率 / n_lies / n_evades / 转化率）\n")
        lines.append("| seed | 提问者 | 主持 | tdhit | dhit | F1 | 复问率 | n_lies | n_evades | 转化率 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for ps in per_seed:
            for qn in q_names:
                for hn in h_names:
                    rows = ps["matrix"][(qn, hn)]["heldout"]
                    cr = cell_mean_opt(rows, "conversion_rate")
                    lines.append(
                        f"| {ps['seed']} | {qn} | {hn} | "
                        f"{cell_mean(rows, 'tdhit'):.3f} | {cell_mean(rows, 'dhit'):.3f} | "
                        f"{cell_mean(rows, 'f1'):.3f} | {cell_mean(rows, 'repeat_rate'):.3f} | "
                        f"{cell_mean(rows, 'n_lies'):.2f} | {cell_mean(rows, 'n_evades'):.2f} | "
                        f"{'N/A' if cr is None else f'{cr:.3f}'} |")

        lines.append("\n### 预注册判据逐种子判定\n")
        lines.append("| seed | E1b F1伤害 trained | random-evader | E1b | "
                     "E1b' tdhit trained | random | E2b 转化率 Q-adv | Q-honest | E2b | "
                     "E2b' F1增益 | 已答复问 Q-adv/Q-honest |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ps in per_seed:
            v = ps["verdict"]
            rq = (f"{v['e2b_reask_ans_qadv']:.3f}" if v["e2b_reask_ans_qadv"] is not None else "N/A",
                  f"{v['e2b_reask_ans_qhonest']:.3f}" if v["e2b_reask_ans_qhonest"] is not None else "N/A")
            lines.append(
                f"| {ps['seed']} | {_fmt_or_na(v['e1b_f1_trained'])} | "
                f"{_fmt_or_na(v['e1b_f1_random'])} | "
                f"{'N/A' if v['e1b_pass'] is None else ('✅' if v['e1b_pass'] else '❌')} | "
                f"{_fmt_or_na(v['e1bp_tdhit_trained'])} | "
                f"{_fmt_or_na(v['e1bp_tdhit_random'])} | "
                f"{_fmt_or_na_conv(v['e2b_conv_qadv'])} | "
                f"{_fmt_or_na_conv(v['e2b_conv_qhonest'])} | "
                f"{'N/A' if v['e2b_pass'] is None else ('✅' if v['e2b_pass'] else '❌')} | "
                f"{v['e2bp_f1_gain']:+.3f} | {rq[0]} / {rq[1]} |")
        n = len(per_seed)
        judged = [ps for ps in per_seed if ps["verdict"]["e1b_pass"] is not None]
        e1_yes = sum(1 for ps in judged if ps["verdict"]["e1b_pass"])
        e2_judged = [ps for ps in per_seed if ps["verdict"]["e2b_pass"] is not None]
        e2_yes = sum(1 for ps in e2_judged if ps["verdict"]["e2b_pass"])
        lines.append(f"\n**ε={eps} 多数裁决：E1b {e1_yes}/{len(judged)} 通过"
                     f"（{len(judged)}/{n} 个 seed 主持实际搪塞，其余处理未激活）；"
                     f"E2b {e2_yes}/{len(e2_judged)} 通过"
                     f"（{len(e2_judged)}/{n} 个 seed 出现被搪塞问句）**\n")
        if any(ps["coev"] for ps in per_seed):
            lines.append("### 共演轨迹（Phase C 逐轮 tdhit，震荡诊断）\n")
            for ps in per_seed:
                if ps["coev"]:
                    lines.append(f"- seed={ps['seed']}: "
                                 + " → ".join(f"{x:.3f}" for x in ps["coev"]))
            lines.append("")

    # ε 区间图（处理激活性 + 判据汇总）
    lines.append("\n## ε 区间图（§3.3：搪塞激活性与「经得起转化」边界）\n")
    lines.append("| ε | 平均搪塞数 trained | 平均谎言数 trained | E1b F1伤害 trained | "
                 "random-evader | E1b 多数裁决 | E2b 转化率 Q-adv | Q-honest | E2b 多数裁决 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for eps, per_seed in runs:
        rows = [r for ps in per_seed
                for r in ps["matrix"][("Q-honest", "trained-evasive")]["heldout"]]
        judged = [ps for ps in per_seed if ps["verdict"]["e1b_pass"] is not None]
        e1_yes = sum(1 for ps in judged if ps["verdict"]["e1b_pass"])
        e2_judged = [ps for ps in per_seed if ps["verdict"]["e2b_pass"] is not None]
        e2_yes = sum(1 for ps in e2_judged if ps["verdict"]["e2b_pass"])
        d_tr = [ps["verdict"]["e1b_f1_trained"] for ps in judged
                if ps["verdict"]["e1b_f1_trained"] is not None]
        d_rn = [ps["verdict"]["e1b_f1_random"] for ps in judged
                if ps["verdict"]["e1b_f1_random"] is not None]
        c_qa = [ps["verdict"]["e2b_conv_qadv"] for ps in per_seed
                if ps["verdict"]["e2b_conv_qadv"] is not None]
        c_qh = [ps["verdict"]["e2b_conv_qhonest"] for ps in per_seed
                if ps["verdict"]["e2b_conv_qhonest"] is not None]
        lines.append(
            f"| {eps} | {cell_mean(rows, 'n_evades'):.2f} | "
            f"{cell_mean(rows, 'n_lies'):.2f} | "
            f"{(sum(d_tr) / len(d_tr)) if d_tr else 'N/A'} | "
            f"{(sum(d_rn) / len(d_rn)) if d_rn else 'N/A'} | "
            f"{e1_yes}/{len(judged)} | "
            f"{(sum(c_qa) / len(c_qa)) if c_qa else 'N/A'} | "
            f"{(sum(c_qh) / len(c_qh)) if c_qh else 'N/A'} | "
            f"{e2_yes}/{len(e2_judged)} |")
    lines.append("")

    lines.append("\n## E3b 混合对局（真实引擎，跑通即过）\n")
    lines.append("| 组合 | vra | yn | dhit | tdhit | F1 | n_evades |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in e3_rows:
        lines.append(f"| {r['mix']} | {r['vra']:.3f} | {r['yn_rate']:.3f} | "
                     f"{r['dhit']:.3f} | {r['tdhit']:.3f} | {r['f1']:.3f} | "
                     f"{r.get('n_evades', 'N/A')} |")
    lines.append("")

    lines.append("\n## 失败模式对照（§6-A pre-declared 六条）\n")
    lines.append("| 模式 | 诊断 | 本 run 观察 |")
    lines.append("|---|---|---|")
    rows = [r for eps, per_seed in runs for ps in per_seed
            for r in ps["matrix"][("Q-honest", "trained-evasive")]["heldout"]]
    lines.append(f"| ①b 主持退化为「有配额就搪塞」 | 搪塞数=k_e 且位置分布均匀 | "
                 f"平均搪塞数 {cell_mean(rows, 'n_evades'):.2f} / 配额 {args.k_e}（跨 ε） |")
    conv_all = [r["conversion_rate"] for r in rows if r["conversion_rate"] is not None]
    lines.append(f"| ②b 提问者永不转化 | 转化率恒 0（对照已答复问率） | "
                 f"平均转化率 {(sum(conv_all) / len(conv_all)) if conv_all else 'N/A'}；"
                 f"已答问句复问率 "
                 f"{cell_mean([r for r in rows if r['reask_answered_rate'] is not None], 'reask_answered_rate'):.3f} |")
    lines.append("| ③b 共演震荡 | Phase C 轨迹循环追逐 | 见各 ε 共演轨迹 |")
    lines.append("| ④b 12 题过拟合 | held-out vs 训练集结论反向 | 双方状态均题目无关，held-out 为准 |")
    lines.append("| ⑤b 信号弱 | Q-adv 与 Q-honest 无差别 | E2b/E2b' 判定即回答 |")
    lines.append("| ⑥b 搪塞完全不激活 | ε≥1.2 区间搪塞数 0 | 见 ε 区间图搪塞数列 |\n")

    lines.append("\n## 训练集对照（held-out 结论的旁证，按 ε 汇总）\n")
    lines.append("| ε | seed | 提问者 | 主持 | tdhit(train) |")
    lines.append("|---|---|---|---|---|")
    for eps, per_seed in runs:
        for ps in per_seed:
            for qn in q_names:
                for hn in h_names:
                    rws = ps["matrix"][(qn, hn)]["train"]
                    lines.append(f"| {eps} | {ps['seed']} | {qn} | {hn} | "
                                 f"{cell_mean(rws, 'tdhit'):.3f} |")
    lines.append(f"\n---\n总耗时 {elapsed:.1f}s。诚实边界：双方均为表格 Q（无函数\n"
                 f"逼近），λ 固定 {LAMBDA_FIXED}（λ×ε 交互是后续）；同源异质/\n"
                 f"跨模型外推不在本阶段范围；E1b/E2b 全负时照实报「搪塞轴未产\n"
                 f"生可学结构」并附 §3.3 经济学解释。\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epsilons", type=float, nargs="*",
                    default=list(EPSILON_SWEEP),
                    help="ε（搪塞成本）扫描区间（§3.3 四档理由）")
    ap.add_argument("--episodes-host", type=int, default=2000)
    ap.add_argument("--episodes-q", type=int, default=4000)
    ap.add_argument("--k", type=int, default=K_DEFAULT, help="每局谎言配额")
    ap.add_argument("--k-e", type=int, default=K_E_DEFAULT, help="每局搪塞配额")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--train-size", type=int, default=8,
                    help="训练题数（12 题题库 → held-out = 12 − 此值）")
    ap.add_argument("--coevolve-rounds", type=int, default=2,
                    help="Phase C 共演轮数（0 = 只交付 Phase H+Q）")
    ap.add_argument("--repeats", type=int, default=10, help="评测时每题重复局数")
    ap.add_argument("--e3-seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("logs", "marl_evade_report.md"))
    args = ap.parse_args()

    puzzles = load_puzzles()
    t0 = time.time()
    print(f"[b1b] 题库 {len(puzzles)} 题 | seeds={args.seeds} | k={args.k} "
          f"| k_e={args.k_e} | λ={LAMBDA_FIXED} | ε 扫描={args.epsilons} | "
          f"H/Q episodes={args.episodes_host}/{args.episodes_q} | "
          f"共演 {args.coevolve_rounds} 轮")

    runs = []
    for eps in args.epsilons:
        per_seed = []
        for seed in args.seeds:
            ts = time.time()
            ps = run_seed(seed, puzzles, args, eps)
            per_seed.append(ps)
            v = ps["verdict"]
            print(f"[ε={eps} seed {seed}] {time.time() - ts:.1f}s | "
                  f"搪塞数 {v['n_evades_trained']:.2f} | "
                  f"谎言数 {v['n_lies_trained']:.2f} | "
                  f"E1b F1伤害 trained {_fmt_or_na(v['e1b_f1_trained'])} vs "
                  f"random-evader {_fmt_or_na(v['e1b_f1_random'])} → "
                  f"{'N/A' if v['e1b_pass'] is None else ('PASS' if v['e1b_pass'] else 'FAIL')} | "
                  f"E2b 转化率 Q-adv {_fmt_or_na_conv(v['e2b_conv_qadv'])} vs "
                  f"Q-honest {_fmt_or_na_conv(v['e2b_conv_qhonest'])} → "
                  f"{'N/A' if v['e2b_pass'] is None else ('PASS' if v['e2b_pass'] else 'FAIL')} | "
                  f"E2b' F1增益 {v['e2bp_f1_gain']:+.3f} | "
                  f"主持状态数 {ps['host_states']}")
        runs.append((eps, per_seed))

    e3_rows = run_e3(puzzles[0], args, args.epsilons[1] if len(args.epsilons) > 1
                     else args.epsilons[0])
    print(f"[e3b] 混合对局 {len(e3_rows)} 组跑通："
          + "；".join(f"{r['mix']} tdhit={r['tdhit']:.3f} n_evades={r.get('n_evades')}"
                     for r in e3_rows))

    write_report(args.out, args, runs, e3_rows, time.time() - t0)
    print(f"[done] 报告已写入 {args.out} | 总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
