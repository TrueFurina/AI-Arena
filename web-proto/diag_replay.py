#!/usr/bin/env python3
"""诊断脚本：重放单模型三角对局，暴露每次调用的真实响应/错误。

背景：test_real_game.py 的降级路径会吞掉 LLMError 详情，无法区分
「模型限流」「prompt 结构问题」「模型能力问题」。本脚本直接调用
players.llm._call，逐轮打印原始返回或真实错误。

零成本：默认模型 zai/glm-4-flash（智谱官方免费）。
用法：TEST_MODEL=zai/glm-4-flash python diag_replay.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_env(path=".env"):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


load_env()

from engine.puzzles import get_puzzle
from players.llm import _call, _transcript_to_msgs


def main():
    model = os.getenv("TEST_MODEL", "zai/glm-4-flash")
    print(f"=== 诊断重放：{model} ===\n")

    puzzle = get_puzzle(None)
    print(f"[题目] {puzzle.title}")
    print(f"[汤面] {puzzle.surface[:80]}...\n")

    transcript = [{"role": "system", "content": f"【汤面】{puzzle.surface}"}]

    q_sys = (
        "你是海龟汤（情境推理）游戏的提问者。你只能通过 yes/no 类问题还原真相。"
        "根据汤面和主持的回答，提出一个能推进推理的具体问题，只输出问题本身，不要解释或寒暄。"
    )
    h_sys = (
        f"你是海龟汤的主持人，掌握完整真相。\n"
        f"汤面：{puzzle.surface}\n"
        f"汤底：{puzzle.solution}\n"
        "规则：对玩家的 yes/no 问题，只能从以下选项作答：是 / 否 / 部分是 / 无关 / 不重要。"
        "严格依据汤底，不得泄露汤底原文或额外信息。"
    )

    for r in range(1, 5):
        # --- 提问者 ---
        conv = [{"role": "user", "content": f"汤面：{puzzle.surface}"}]
        conv += _transcript_to_msgs(transcript, "diag-q")
        t0 = time.time()
        try:
            q = _call(model, q_sys, conv, max_tokens=120)
            dt = time.time() - t0
            print(f"[R{r} 提问] ({dt:.1f}s) {q!r}")
        except Exception as e:
            dt = time.time() - t0
            print(f"[R{r} 提问] ({dt:.1f}s) ❌ {type(e).__name__}: {e}")
            q = "（降级）这件事发生在室内还是室外？"
        transcript.append({"round": r, "speaker": "diag-q", "role": "question", "content": q, "ts": 0})

        time.sleep(1)

        # --- 主持 ---
        h_conv = _transcript_to_msgs(transcript, "diag-host")
        h_conv.append({"role": "user", "content": f"玩家问：{q}"})
        t0 = time.time()
        try:
            a = _call(model, h_sys, h_conv, temperature=0.2, max_tokens=40)
            dt = time.time() - t0
            print(f"[R{r} 主持] ({dt:.1f}s) {a!r}")
        except Exception as e:
            dt = time.time() - t0
            print(f"[R{r} 主持] ({dt:.1f}s) ❌ {type(e).__name__}: {e}")
            a = "不重要"
        transcript.append({"round": r, "speaker": "diag-host", "role": "answer", "content": a, "ts": 0})
        time.sleep(1)

    # --- 终局还原 ---
    conv = [{"role": "user", "content": f"汤面：{puzzle.surface}"}]
    conv += _transcript_to_msgs(transcript, "diag-q")
    conv.append({"role": "user", "content": "现在请写出你的最终还原。"})
    t0 = time.time()
    try:
        g = _call(model, "你是海龟汤提问者。请基于全部问答，写出你对该事件完整真相的还原（一段连贯文字）。",
                  conv, temperature=0.3, max_tokens=500)
        print(f"[还原] ({time.time()-t0:.1f}s) len={len(g)} | {g[:60]}...")
    except Exception as e:
        print(f"[还原] ❌ {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
