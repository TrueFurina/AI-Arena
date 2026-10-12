#!/usr/bin/env python3
"""L2 执行层 · RL 提问策略训练（零成本，不调用任何付费 API）。

环境：MockHost（golden_qa 子串匹配）+ 本地题库，无网络、无 API、无费用。
产出：data/rl_policy.json（Q 表）+ 训练曲线 + 贪心 vs 随机对比。

用法：
  python train_rl.py                 # 默认 400 局
  python train_rl.py --episodes 800  # 自定义局数

为什么必须有「随机基线对比」：
  单看训练曲线上升证明不了策略学到了东西（可能只是探索噪声）。只有
  在同一题库、同一奖励函数、固定种子下，贪心策略显著高于随机策略，
  才能说明 Q 表确实编码了有效信息。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.puzzles import load_puzzles
from players.mock import MockHost
from players.rl import (
    build_action_bank, RLQuestioner, TabularQPolicy, POLICY_DEFAULT_PATH, run_episode,
)


def evaluate(bank, informative, policy, puzzles, repeats=20, seed=0,
             explore=False, max_rounds=8):
    """在全部题目上评估策略，返回 (平均累计奖励, 平均 F1)。"""
    host = MockHost()
    q = RLQuestioner(policy=policy, action_bank=bank, informative=informative,
                     explore=explore)
    rewards, f1s = [], []
    for _ in range(repeats):
        for p in puzzles:
            out = run_episode(p, host, q, max_rounds=max_rounds, train=False)
            rewards.append(out["reward"])
            f1s.append(out["f1"])
    return sum(rewards) / len(rewards), sum(f1s) / len(f1s)


def informative_gap(policy, bank, informative, puzzles):
    """有信息型动作与无信息型动作的平均 Q 差（题目条件下的空状态）。

    早期版本（状态不含题目）该值正负随种子乱跳——POMDP 表示缺陷所致。
    这里逐题统计，因为 Q 现在是条件于题目的。
    """
    gaps = []
    for p in puzzles:
        row = policy.Q.get(f"{p.soup_id}|", None)
        if row is None:
            continue
        pos = [row[i] for i in informative]
        neg = [row[i] for i in range(len(bank)) if i not in informative]
        if pos and neg:
            gaps.append(sum(pos) / len(pos) - sum(neg) / len(neg))
    return (sum(gaps) / len(gaps)) if gaps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=POLICY_DEFAULT_PATH)
    args = ap.parse_args()

    puzzles = load_puzzles()
    bank, informative = build_action_bank(puzzles)
    host = MockHost()
    policy = TabularQPolicy(len(bank), seed=args.seed)
    q = RLQuestioner(policy=policy, action_bank=bank, informative=informative,
                     explore=True)

    print(f"[train] 题库 {len(puzzles)} 题 | 动作库 {len(bank)} 个 "
          f"（有信息型 {len(informative)} / 无信息型 {len(bank) - len(informative)}）")
    print(f"        状态 = 题目ID|已问集合 | episodes={args.episodes} "
          f"rounds={args.max_rounds} seed={args.seed}")

    t0 = time.time()
    curve = []
    for ep in range(1, args.episodes + 1):
        p = puzzles[(ep - 1) % len(puzzles)]
        out = run_episode(p, host, q, max_rounds=args.max_rounds, train=True)
        curve.append(out["reward"])
        if ep % 100 == 0:
            recent = sum(curve[-50:]) / 50
            print(f"  ep {ep:4d} · 近50局平均奖励 {recent:+.3f}")

    # ---- 训练后：贪心 vs 随机（同题库、同奖励、固定种子）----
    greedy_r, greedy_f1 = evaluate(bank, informative, policy, puzzles, explore=False,
                                   seed=args.seed, max_rounds=args.max_rounds)
    random_policy = TabularQPolicy(len(bank), epsilon=1.0, seed=args.seed)
    rand_r, rand_f1 = evaluate(bank, informative, random_policy, puzzles, explore=True,
                               seed=args.seed, max_rounds=args.max_rounds)

    policy.save(args.out)

    print("\n=== 训练结果（贪心 vs 随机，固定种子）===")
    print(f"  贪心策略：平均累计奖励 {greedy_r:+.3f} | 平均还原 F1 {greedy_f1:.3f}")
    print(f"  随机策略：平均累计奖励 {rand_r:+.3f} | 平均还原 F1 {rand_f1:.3f}")
    print(f"  奖励差：{greedy_r - rand_r:+.3f}（>0 说明 Q 表学到了有效信息）")
    print(f"  F1  差：{greedy_f1 - rand_f1:+.3f}")

    gap = informative_gap(policy, bank, informative, puzzles)
    print(f"  有信息型-无信息型 平均 Q 差（题目条件下）：{gap:+.3f} "
          f"→ {'已学会偏好有信息动作' if gap > 0 else '未分离'}")
    print(f"\n[done] 策略已保存 {args.out} | Q 状态数 {len(policy.Q)} | "
          f"耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
