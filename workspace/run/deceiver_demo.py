"""M5 骗子变体 · 零 key 可运行演示（PoC）。

证明「欺骗维度」能落地并产出新指标，不依赖任何 API key。

做了什么：
- 同一提问者（MockQuestioner）分别在「诚实 GM (SmartGM)」与
  「撒谎 GM (DeceiverGM, lie_rate=0.3)」下跑同一套题；
- 比较两种环境下的还原度，算出 **欺骗鲁棒性** = 撒谎环境还原度 / 诚实环境还原度；
- 打印每题还原度对比 + 整体鲁棒性。

为什么是 PoC 而非终态（见 specs/deceiver-design.md）：
- 真正的 M5 还要引入「提问者主动检测撒谎」的动作协议（challenge）与
  lie_detection_rate 指标；本脚本先验证引擎层（GM 撒谎、Game 记录 lied、
  metrics 聚合 lie_rate）已打通。

用法：
    python run/deceiver_demo.py                 # 默认 lie_rate=0.3, seed=42
    python run/deceiver_demo.py --lie-rate 0.5 --seed 7
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import Game
from engine.puzzles import PUZZLES
from players.gm import SmartGM, DeceiverGM
from players.mock_player import MockQuestioner
from eval.metrics import deception_robustness, mean_lie_rate


def run_one(puzzle, gm, max_turns=20):
    q = MockQuestioner(key_facts=puzzle["key_facts"], bottom=puzzle["bottom"])
    game = Game(puzzle, gm, q, max_turns=max_turns)
    log = game.run()
    r = log["result"]
    return r["restoration"], r["deception"]["lie_rate"]


def main():
    parser = argparse.ArgumentParser(description="M5 骗子变体零 key 演示")
    parser.add_argument("--lie-rate", type=float, default=0.3, help="GM 撒谎概率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    parser.add_argument("--max-turns", type=int, default=20)
    args = parser.parse_args()

    print(f"M5 骗子变体演示 · lie_rate={args.lie_rate} · seed={args.seed}\n")

    truth_res, lie_res, lie_rates = [], [], []
    header = f"{'题目':<12} | {'诚实还原度':>8} | {'撒谎还原度':>8} | {'鲁棒性':>6} | {'实际撒谎率':>8}"
    print(header)
    print("-" * len(header))

    for p in PUZZLES:
        r_truth, _ = run_one(p, SmartGM(), args.max_turns)
        r_lie, actual_lr = run_one(
            p, DeceiverGM(lie_rate=args.lie_rate, rng=__import__("random").Random(args.seed)),
            args.max_turns,
        )
        rob = deception_robustness(r_truth, r_lie)
        truth_res.append(r_truth)
        lie_res.append(r_lie)
        lie_rates.append(actual_lr)
        print(f"{p['id']:<12} | {r_truth:>8.2f} | {r_lie:>8.2f} | {rob:>6.2f} | {actual_lr:>8.2f}")

    print("-" * len(header))
    avg_truth = sum(truth_res) / len(truth_res)
    avg_lie = sum(lie_res) / len(lie_res)
    avg_rob = deception_robustness(avg_truth, avg_lie)
    print(f"{'平均':<12} | {avg_truth:>8.2f} | {avg_lie:>8.2f} | {avg_rob:>6.2f} | {mean_lie_rate([{'lie_rate': x} for x in lie_rates]):>8.2f}")

    print("\n解读：")
    print(f"- 鲁棒性=1 表示 GM 撒谎完全不影响推理；越接近 0 越脆弱。")
    print(f"- 当前提问者是无策略的 mock（照念关键事实），鲁棒性偏低属预期；")
    print(f"  真实模型若具备「矛盾检测」，鲁棒性应显著高于此基线。")


if __name__ == "__main__":
    main()
