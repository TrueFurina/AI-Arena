#!/usr/bin/env python3
"""
跑一局海龟汤对战，并落 JSON 日志 + 评测报告。

用法示例：
  # 纯 Mock 离线跑通（默认，零依赖）
  python run_game.py

  # 指定题目 + 两个真实模型对战（需 .env 配好 key，且 litellm 已装）
  python run_game.py --puzzle turtle_soup_001 \
      --questioners zai/glm-4.5-flash,deepseek/deepseek-chat \
      --host zai/glm-4.7 --judge openai/gpt-4o-mini --rounds 6

  # 接入免费 ashbwin 端点（端点需可达；见 arena_providers.py）
  python run_game.py --questioners ashbwin,mock-gpt4o --host ashbwin
"""
from __future__ import annotations

import argparse
import os
import sys

# 让脚本可直接从 AI-Arena 根目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from engine.game import TurtleSoupGame
from engine.puzzles import get_puzzle
from players import make_player
from analysis.report import build_report


def main():
    ap = argparse.ArgumentParser(description="AI Arena 海龟汤单局运行器")
    ap.add_argument("--puzzle", default=None, help="题目 id，缺省取第一题")
    ap.add_argument("--questioners", default="mock-gpt4o,mock-glm", help="逗号分隔的提问者模型名")
    ap.add_argument("--host", default="mock-host", help="主持模型名")
    ap.add_argument("--judge", default="mock", help="裁判模型名（mock 或某模型）")
    ap.add_argument("--rounds", type=int, default=8, help="最大提问轮数")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
                    help="日志/报告输出目录")
    args = ap.parse_args()

    puzzle = get_puzzle(args.puzzle)
    qspecs = [s.strip() for s in args.questioners.split(",") if s.strip()]
    questioners = [make_player(spec, role="questioner", index=i) for i, spec in enumerate(qspecs)]
    host = make_player(args.host, role="host")
    judge = make_player(args.judge, role="judge") if args.judge != "mock" else make_player("mock-judge", role="judge")

    print(f"[run] 题目: {puzzle.title} ({puzzle.soup_id})")
    print(f"[run] 提问者: {[q.model for q in questioners]} | 主持: {host.model} | 裁判: {judge.model}")
    print(f"[run] 轮数: {args.rounds}\n")

    game = TurtleSoupGame(puzzle, host, questioners, judge=judge, max_rounds=args.rounds)
    result = game.run()

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"game_{result.game_id}.json")
    result.save(log_path)
    print(f"[run] 对局日志已保存: {log_path}")

    report_path = build_report(result, args.out)
    if report_path:
        print(f"[run] 评测报告图: {report_path}")

    # 控制台摘要
    print("\n========== 评分 ==========")
    for s in result.scores:
        print(f"  {s['player']:>14} | {s['score']:>5}  | {s['rationale'][:50]}")
    print(f"  胜者: {result.winner or '（无）'}")


if __name__ == "__main__":
    main()
