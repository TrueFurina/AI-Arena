#!/usr/bin/env python3
"""L2 泛化能力评测：held-out 协议 + 样本阶梯（零成本，不调用付费 API）。

要回答的问题只有一个：**这个 RL 提问策略在没见过的新题上还管用吗？**
训练题上分数高说明不了泛化——把题目 ID 塞进状态后，策略只要记住
「这道题问哪几个词」就能拿高分，那是记忆不是能力。

协议设计（三条，缺一不可）：
  ① held-out 划分：按 seed 打乱题库，前 N 题训练、其余测试。测试题在
     训练过程中**一次都不出现**。
  ② 动作库按「词汇表」处理：从**全题库**派生，不算泄漏——它只是候选
     提问用词，不含任何答案信息，策略仍需自己学哪个词有用。（若只从
     训练题派生，测的就是动作空间覆盖率而非策略泛化，答非所问。）
  ③ 两种状态表示对照：
       puzzle  题目条件（state 含 soup_id）→ 特化，预期训练题强、
               新题上状态键不存在 → Q 全零 → 退化
       global  题目无关（state 只含已问集合）→ 可迁移，预期训练题弱、
               但新题上仍能复用学到的偏好

样本阶梯（用户的铁律：单一样本量的定性结论不可信）：
  训练题数 3 / 6 / 9 各跑一遍，看新题表现是否随训练题增多而改善。
  若改善 → 是数据量问题；若不改善 → 是动作抽象本身的问题。

指标（分开报，别混）：
  hit_rate 线索命中率——提问质量的直接度量，不受汤底措辞影响（主指标）
  f1       还原质量——会被「问中了但汤底用词不同」稀释（次指标）
  reward   环境奖励——代理指标，含「不重复」等取巧成分
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.puzzles import load_puzzles
from engine.evaluate import hit_rate
from players.mock import MockHost
from players.rl import (
    build_action_bank, RLQuestioner, TabularQPolicy, run_episode,
)

MODES = ("puzzle", "global")
TRAIN_SIZES = (3, 6, 9)


def split_puzzles(puzzles, train_size: int, seed: int):
    rng = random.Random(seed)
    ps = list(puzzles)
    rng.shuffle(ps)
    return ps[:train_size], ps[train_size:]


def train_policy(train_set, bank, info, mode, episodes, seed, max_rounds):
    host = MockHost()
    policy = TabularQPolicy(len(bank), seed=seed)
    q = RLQuestioner(policy=policy, action_bank=bank, informative=info,
                     explore=True, use_puzzle_state=(mode == "puzzle"))
    for ep in range(episodes):
        run_episode(train_set[ep % len(train_set)], host, q,
                    max_rounds=max_rounds, train=True)
    return policy


def evaluate_policy(puzzles, bank, info, policy, mode, repeats, max_rounds):
    host = MockHost()
    q = RLQuestioner(policy=policy, action_bank=bank, informative=info,
                     explore=False, use_puzzle_state=(mode == "puzzle"))
    rewards, hits, f1s = [], [], []
    for _ in range(repeats):
        for p in puzzles:
            out = run_episode(p, host, q, max_rounds=max_rounds, train=False)
            answers = [m["content"] for m in out["transcript"]
                       if m.get("role") == "answer"]
            rewards.append(out["reward"])
            hits.append(hit_rate(answers))
            f1s.append(out["f1"])
    n = len(rewards)
    return sum(rewards) / n, sum(hits) / n, sum(f1s) / n


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
    args = ap.parse_args()

    puzzles = load_puzzles()
    bank, info = build_action_bank(puzzles)  # 词汇表：全题库派生，见协议②
    print(f"[gen] 题库 {len(puzzles)} 题 | 动作库(词汇表) {len(bank)} 个 | "
          f"每题 {args.per_puzzle_episodes} 局 | rounds={args.max_rounds} "
          f"seed={args.seed}")

    t0 = time.time()
    rows = []
    for n_train in args.train_sizes:
        train_set, heldout = split_puzzles(puzzles, n_train, args.seed)
        if not heldout:
            print(f"  训练题数 {n_train}：无 held-out 题，跳过")
            continue
        episodes = args.per_puzzle_episodes * n_train
        for mode in MODES:
            pol = train_policy(train_set, bank, info, mode, episodes,
                               args.seed, args.max_rounds)
            tr_r, tr_h, tr_f = evaluate_policy(train_set, bank, info, pol, mode,
                                               args.repeats, args.max_rounds)
            ho_r, ho_h, ho_f = evaluate_policy(heldout, bank, info, pol, mode,
                                               args.repeats, args.max_rounds)
            # 未训练基线：新题上 Q 全零 → 贪心退化为固定取 index 0。若 held-out
            # 恰好含该题，会「命中」，那是假泛化，必须用基线扣除。
            base = TabularQPolicy(len(bank), seed=args.seed)
            _, bh_h, bh_f = evaluate_policy(heldout, bank, info, base, mode,
                                            args.repeats, args.max_rounds)
            rows.append({
                "n_train": n_train, "n_held": len(heldout), "mode": mode,
                "train_hit": tr_h, "held_hit": ho_h, "baseline_hit": bh_h,
                "train_f1": tr_f, "held_f1": ho_f, "baseline_f1": bh_f,
                "train_reward": tr_r, "held_reward": ho_r,
            })

    print("\n=== 泛化评测（held-out 从未参与训练，每题训练量固定）===")
    print("训练题 新题数 模式     | 命中率 训练/新题/未训练基线 | F1 训练/新题 | "
          "奖励 训练/新题")
    for r in rows:
        print(f"  {r['n_train']:>2}    {r['n_held']:>2}   {r['mode']:<7} | "
              f"{r['train_hit']:.3f} / {r['held_hit']:.3f} / {r['baseline_hit']:.3f}   | "
              f"{r['train_f1']:.3f} / {r['held_f1']:.3f}   | "
              f"{r['train_reward']:+.2f} / {r['held_reward']:+.2f}")

    print("\n=== 结论口径（新题 - 未训练基线 = 真泛化增益）===")
    for mode in MODES:
        mr = sorted([r for r in rows if r["mode"] == mode],
                    key=lambda x: x["n_train"])
        if not mr:
            continue
        gains = [r["held_hit"] - r["baseline_hit"] for r in mr]
        print(f"  {mode:<7} 真泛化增益 = "
              f"{' / '.join(f'{g:+.3f}' for g in gains)}   "
              f"(训练题数 {'→'.join(str(r['n_train']) for r in mr)})")
        print(f"          训练题命中率 = "
              f"{' / '.join(f'{r['train_hit']:.3f}' for r in mr)}；"
              f"落差 = {' / '.join(f'{r['train_hit'] - r['held_hit']:+.3f}' for r in mr)}")

    out = os.path.join("logs", "generalization_report.md")
    os.makedirs("logs", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# L2 泛化能力评测（held-out 协议）\n\n")
        f.write(f"- 题库 {len(puzzles)} 题，动作库（词汇表）{len(bank)} 个\n")
        f.write(f"- 每题训练 {args.per_puzzle_episodes} 局（阶梯间按题固定，"
                f"消除训练量混淆）, rounds={args.max_rounds}, "
                f"repeats={args.repeats}, seed={args.seed}\n\n")
        f.write("| 训练题 | 新题数 | 模式 | 命中率 训练/新题/基线 | F1 训练/新题 | "
                "奖励 训练/新题 |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['n_train']} | {r['n_held']} | {r['mode']} | "
                    f"{r['train_hit']:.3f} / {r['held_hit']:.3f} / "
                    f"{r['baseline_hit']:.3f} | "
                    f"{r['train_f1']:.3f} / {r['held_f1']:.3f} | "
                    f"{r['train_reward']:+.2f} / {r['held_reward']:+.2f} |\n")
    print(f"\n[done] 报告已写入 {out} | 耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
