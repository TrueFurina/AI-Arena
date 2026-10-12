#!/usr/bin/env python3
"""
AI Arena · 真实模型端到端对战验证（最小依赖版）
==============================================
用真实模型（默认 zai/glm-4-flash）跑完整一局：提问者+主持+裁判全用真模型，
证明 engine + players(LLM) + 评分 在真实模型下端到端可用（与 ashbwin 无关）。

只依赖 litellm（不依赖 matplotlib/pandas），手动加载 .env 的 key 到环境变量。
用法：/tmp/venv/bin/python test_real_game.py
"""
import os
import sys

# 手动加载 .env 到环境变量（不依赖 python-dotenv）
def _load_one(path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                # 去掉行内注释（首个 # 之后），再剥引号与首尾空白——否则含中文的
                # 注释会被塞进 Authorization 头，触发 latin-1/ascii 编码错误并掩盖真实 auth 报错
                v = v.split("#", 1)[0].strip().strip('"').strip("'")
                if k:
                    os.environ.setdefault(k, v)
    except FileNotFoundError:
        pass


def load_env():
    # 本地 .env 优先；workspace/.env 作为兜底（deepseek/qwen 等 key 按项目布局放在该线）
    here = os.path.dirname(os.path.abspath(__file__))
    _load_one(os.path.join(here, ".env"))
    _load_one(os.path.join(here, "..", "workspace", ".env"))


load_env()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.game import TurtleSoupGame
from engine.puzzles import get_puzzle
from players import make_player


def main():
    model = os.getenv("TEST_MODEL", "zai/glm-4-flash")
    print(f"=== 真实模型端到端对战：{model} ===\n")

    puzzle = get_puzzle(None)
    questioner = make_player(model, role="questioner", index=0)
    host = make_player(model, role="host")
    judge = make_player(model, role="judge")

    game = TurtleSoupGame(puzzle, host, [questioner], judge=judge, max_rounds=4)
    result = game.run()

    print(f"[题目] {puzzle.title}")
    print(f"[轮数] {len([m for m in result.transcript if m['role']=='question'])} 次提问\n")
    print("--- 对局实录（节选）---")
    for m in result.transcript[:10]:
        who = {"question": "提问者", "answer": "主持", "guess": "还原", "system": "系统"}.get(m["role"], m["role"])
        print(f"  {who}: {m['content'][:80]}")
    if len(result.transcript) > 10:
        print(f"  …（共 {len(result.transcript)} 条）")

    print("\n--- 评分 ---")
    for s in result.scores:
        print(f"  {s['player']:>14} | {s['score']:>5} | {s['rationale'][:60]}")
    print(f"  胜者: {result.winner or '（无）'}")
    print("\n✅ 真实模型端到端对战成功：引擎、LLM 提问者/主持/裁判、评分均工作。")


if __name__ == "__main__":
    main()
