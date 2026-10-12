#!/usr/bin/env python3
"""L2 泛化能力评测：held-out 协议 + 样本阶梯（零成本，不调用付费 API）。

要回答的问题只有一个：**这个 RL 提问策略在没见过的新题上还管用吗？**
训练题上分数高说明不了泛化——把题目 ID 塞进状态后，策略只要记住
「这道题问哪几个词」就能拿高分，那是记忆不是能力。

协议设计（四条，缺一不可）：
  ① held-out 划分：按 seed 打乱题库，前 N 题训练、其余测试。测试题在
     训练过程中**一次都不出现**。
  ② 动作库按「词汇表」处理：从**全题库**派生，不算泄漏——它只是候选
     提问用词，不含任何答案信息，策略仍需自己学哪个词有用。（若只从
     训练题派生，测的就是动作空间覆盖率而非策略泛化，答非所问。）
  ③ 三种动作/状态配置对照：
       puzzle    关键词动作 + 题目条件状态（含 soup_id）→ 特化，新题上
                 状态键不存在 → Q 全零 → 退化
       global    关键词动作 + 题目无关状态 → 状态可共享，但动作仍与题绑定
       semantic  语义类别动作 + 题目无关状态 → 动作与状态都跨题通用，
                 这是唯一存在迁移路径的配置（engine/semantic.py）
  ④ 新题收益必须扣基线：未训练策略在新题上可能「碰巧问对」（关键词模式
     下曾出现策略一直问 bank[0]，而该题恰好答「是」）。另加**随机不重复**
     基线——它扣除「仅靠不重复提问就能拿到的分」，剩下的才是真学到的东西。

样本阶梯（用户的铁律：单一样本量的定性结论不可信）：
  训练题数 3 / 6 / 9 各跑一遍，看新题表现是否随训练题增多而改善。
  若改善 → 是数据量问题；若不改善 → 是动作抽象本身的问题。

指标（分开报，别混）：
  dhit     去重线索命中率——按不同问句去重（**主指标**，堵住重复刷分）
  hit_rate 原始命中率——不惩罚重复，语义模式下会被重复提问刷爆，仅参考
  f1       还原质量——⚠️ 语义模式下与关键词模式**不可比**（见下）
  reward   环境奖励——代理指标，含「不重复」等取巧成分

⚠️ F1 在 semantic 模式下不可比：语义动作的还原句由类别名拼成（「与交通
工具有关」），不含汤底实体词，F1 天然很低。这不是 bug，是语义抽象牺牲
还原换迁移的代价（详见 engine/semantic.py 模块文档）。semantic 模式看 dhit。
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.puzzles import load_puzzles
from engine.evaluate import distinct_hit_rate, hit_rate
from engine.semantic import CATEGORY_NAMES, CategoryHost, category_answer
from players.mock import MockHost
from players.rl import (
    build_action_bank, RLQuestioner, TabularQPolicy, run_episode,
)

MODES = ("puzzle", "global", "semantic")
TRAIN_SIZES = (3, 6, 9)


def split_puzzles(puzzles, train_size: int, seed: int):
    rng = random.Random(seed)
    ps = list(puzzles)
    rng.shuffle(ps)
    return ps[:train_size], ps[train_size:]


def _parse_asked(state) -> set:
    """从状态串里解析已问动作（global/semantic 模式下 state = "0,3,5"）。"""
    if isinstance(state, str):
        tail = state.split("|", 1)[-1]  # 兼容 puzzle 模式的「题ID|...」
        return {int(x) for x in tail.split(",") if x.strip().lstrip("-").isdigit()}
    return set(state)


class DistinctRandomPolicy:
    """在未问过的动作里均匀随机——扣除「仅靠不重复就能拿到的分」。

    为什么需要它：不重复提问本身就能提高去重命中率（重复问不增加分子）。
    若只跟「Q 全零的贪心」比，会把「学会了别重复问」误算成「学会了问什么」。
    """

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.gamma = 0.9  # run_episode 的 MC 回传会读该字段（train=False 时不用）
        self._rng = random.Random(seed)

    def select_action(self, state, explore: bool = True) -> int:
        asked = _parse_asked(state)
        rest = [i for i in range(self.n_actions) if i not in asked]
        return self._rng.choice(rest or list(range(self.n_actions)))

    def greedy_action(self, state):
        return self.select_action(state)

    def update(self, *a, **k):
        pass


class FixedOrderPolicy:
    """按固定顺序问（用于「先验上限」参考）。

    顺序由**训练集**的类别频率决定。若训练出的 RL 策略在新题上≈这个参考值，
    说明它学到的就是「全局先验」，没有更多——这是必须说清楚的一件事。
    """

    def __init__(self, order):
        self.order = list(order)
        self.gamma = 0.9

    def select_action(self, state, explore: bool = True) -> int:
        asked = _parse_asked(state)
        for a in self.order:
            if a not in asked:
                return a
        return self.order[0]

    def greedy_action(self, state):
        return self.select_action(state)

    def update(self, *a, **k):
        pass


def make_env(mode, puzzles):
    """按模式装配 (动作库, 有信息动作集合, 主持人, 是否含题目状态)。"""
    if mode == "semantic":
        bank = list(CATEGORY_NAMES)
        return bank, set(), CategoryHost(), False
    bank, info = build_action_bank(puzzles)  # 词汇表：全题库派生，见协议②
    return bank, info, MockHost(), (mode == "puzzle")


def make_questioner(bank, info, policy, use_puzzle_state, explore=False):
    return RLQuestioner(policy=policy, action_bank=bank, informative=info,
                        explore=explore, use_puzzle_state=use_puzzle_state)


def train_policy(train_set, env, episodes, seed, max_rounds):
    bank, info, host, use_ps = env
    policy = TabularQPolicy(len(bank), seed=seed)
    q = make_questioner(bank, info, policy, use_ps, explore=True)
    for ep in range(episodes):
        run_episode(train_set[ep % len(train_set)], host, q,
                    max_rounds=max_rounds, train=True)
    return policy


def evaluate_policy(puzzles, env, policy, repeats, max_rounds):
    bank, info, host, use_ps = env
    q = make_questioner(bank, info, policy, use_ps, explore=False)
    rewards, hits, dhits, f1s = [], [], [], []
    for _ in range(repeats):
        for p in puzzles:
            out = run_episode(p, host, q, max_rounds=max_rounds, train=False)
            answers = [m["content"] for m in out["transcript"]
                       if m.get("role") == "answer"]
            rewards.append(out["reward"])
            hits.append(hit_rate(answers))
            dhits.append(distinct_hit_rate(out["transcript"]))
            f1s.append(out["f1"])
    n = len(rewards)
    return (sum(rewards) / n, sum(hits) / n, sum(dhits) / n, sum(f1s) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-puzzle-episodes", type=int, default=800,
                    help="**每题**训练局数（非总局数）。阶梯实验中必须按题固定，"
                         "否则训练题越多、每题分到的训练量越少，会把「训练不足」"
                         "误读成「泛化差」")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-sizes", type=int, nargs="*", default=list(TRAIN_SIZES))
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    args = ap.parse_args()

    puzzles = load_puzzles()
    print(f"[gen] 题库 {len(puzzles)} 题 | 每题 {args.per_puzzle_episodes} 局 | "
          f"rounds={args.max_rounds} repeats={args.repeats} seed={args.seed}")

    t0 = time.time()
    rows = []
    for n_train in args.train_sizes:
        train_set, heldout = split_puzzles(puzzles, n_train, args.seed)
        if not heldout:
            print(f"  训练题数 {n_train}：无 held-out 题，跳过")
            continue
        episodes = args.per_puzzle_episodes * n_train
        for mode in args.modes:
            env = make_env(mode, puzzles)
            bank = env[0]
            pol = train_policy(train_set, env, episodes, args.seed, args.max_rounds)
            tr_r, tr_h, tr_d, tr_f = evaluate_policy(
                train_set, env, pol, args.repeats, args.max_rounds)
            ho_r, ho_h, ho_d, ho_f = evaluate_policy(
                heldout, env, pol, args.repeats, args.max_rounds)

            # 基线 1：未训练（Q 全零贪心）——新题上的「碰巧问对」
            base = TabularQPolicy(len(bank), seed=args.seed)
            _, bh_h, bh_d, _ = evaluate_policy(
                heldout, env, base, args.repeats, args.max_rounds)
            # 基线 2：随机不重复——扣除「仅靠不重复」就能拿到的分
            rnd = DistinctRandomPolicy(len(bank), seed=args.seed)
            _, rh_h, rh_d, _ = evaluate_policy(
                heldout, env, rnd, args.repeats, args.max_rounds)
            # 参考：先验上限（仅 semantic 有意义，顺序由训练集频率决定）
            prior_d = None
            if mode == "semantic":
                order = _prior_order(train_set, bank)
                _, _, prior_d, _ = evaluate_policy(
                    heldout, env, FixedOrderPolicy(order),
                    args.repeats, args.max_rounds)

            rows.append({
                "n_train": n_train, "n_held": len(heldout), "mode": mode,
                "train_dhit": tr_d, "held_dhit": ho_d,
                "untrained_dhit": bh_d, "random_dhit": rh_d,
                "prior_dhit": prior_d,
                "train_hit": tr_h, "held_hit": ho_h,
                "train_f1": tr_f, "held_f1": ho_f,
                "train_reward": tr_r, "held_reward": ho_r,
            })

    print("\n=== 泛化评测（held-out 从未参与训练；dhit = 去重线索命中率，主指标）===")
    print("训练题 模式      | dhit 训练/新题/未训练/随机不重复/先验上限 | "
          "F1 训练/新题")
    for r in rows:
        prior = f"{r['prior_dhit']:.3f}" if r["prior_dhit"] is not None else "  -  "
        print(f"  {r['n_train']:>2}   {r['mode']:<8} | "
              f"{r['train_dhit']:.3f} / {r['held_dhit']:.3f} / "
              f"{r['untrained_dhit']:.3f} / {r['random_dhit']:.3f} / {prior}   | "
              f"{r['train_f1']:.3f} / {r['held_f1']:.3f}")

    print("\n=== 真泛化增益（新题 - 随机不重复基线）===")
    for mode in args.modes:
        mr = sorted([r for r in rows if r["mode"] == mode],
                    key=lambda x: x["n_train"])
        if not mr:
            continue
        g = [r["held_dhit"] - r["random_dhit"] for r in mr]
        print(f"  {mode:<8} dhit 增益 = {' / '.join(f'{x:+.3f}' for x in g)}   "
              f"(训练题数 {'→'.join(str(r['n_train']) for r in mr)})")
        print(f"           新题 dhit   = "
              f"{' / '.join(f'{r['held_dhit']:.3f}' for r in mr)}；"
              f"落差(训练-新题) = "
              f"{' / '.join(f'{r['train_dhit'] - r['held_dhit']:+.3f}' for r in mr)}")
        if mr[0]["prior_dhit"] is not None:
            print("           先验上限   = "
                  f"{' / '.join(f'{r['prior_dhit']:.3f}' for r in mr)}"
                  "  ← 训练值贴近它则说明只学到全局先验")

    out = os.path.join("logs", "generalization_report.md")
    os.makedirs("logs", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# L2 泛化能力评测（held-out 协议）\n\n")
        f.write(f"- 题库 {len(puzzles)} 题；每题训练 {args.per_puzzle_episodes} 局"
                f"（阶梯间按题固定，消除训练量混淆）；rounds={args.max_rounds}, "
                f"repeats={args.repeats}, seed={args.seed}\n")
        f.write("- 主指标 dhit = 去重线索命中率（按不同问句去重，堵住重复提问刷分）\n")
        f.write("- ⚠️ semantic 模式的 F1 与关键词模式**不可比**：语义动作还原句"
                f"由类别名拼成，不含汤底实体词\n\n")
        f.write("| 训练题 | 模式 | dhit 训练/新题/未训练/随机不重复/先验上限 | "
                "F1 训练/新题 | 奖励 训练/新题 |\n|---|---|---|---|---|\n")
        for r in rows:
            prior = f"{r['prior_dhit']:.3f}" if r["prior_dhit"] is not None else "-"
            f.write(f"| {r['n_train']} | {r['mode']} | "
                    f"{r['train_dhit']:.3f} / {r['held_dhit']:.3f} / "
                    f"{r['untrained_dhit']:.3f} / {r['random_dhit']:.3f} / {prior} | "
                    f"{r['train_f1']:.3f} / {r['held_f1']:.3f} | "
                    f"{r['train_reward']:+.2f} / {r['held_reward']:+.2f} |\n")
    print(f"\n[done] 报告已写入 {out} | 耗时 {time.time() - t0:.1f}s")


def _prior_order(train_set, bank):
    """按训练集上的类别「是」频率降序排动作（全局先验，不含新题信息）。"""
    freq = []
    for i, cat in enumerate(bank):
        n = sum(1 for p in train_set if category_answer(p.solution, cat) == "是")
        freq.append((n, i))
    freq.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in freq]


if __name__ == "__main__":
    main()
