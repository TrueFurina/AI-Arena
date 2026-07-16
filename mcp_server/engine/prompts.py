"""提示词构建：玩家回合、裁判应答、赛后复盘。

这些函数既供 LLMPlayer 内部调用，也供 server.py 的 @mcp.prompt() 复用，
保证"引擎里用的 prompt"和"暴露给智能体的 prompt 模板"完全一致。
"""
from __future__ import annotations

from typing import Any


def build_player_prompt(state: dict[str, Any], player_name: str) -> str:
    """构造某玩家当前回合的系统+用户提示。

    关键约束：只给 surface + 历史对话，绝不泄露 truth / key_facts。
    """
    lines = [
        "你正在玩海龟汤（情境推理游戏）。你是玩家，目标是用 yes/no 提问逐步还原真相，"
        "或在有把握时直接提交对完整故事的猜想。",
        "",
        f"你的身份：{player_name}",
        f"题目（汤面）：{state['surface']}",
        "",
        "已发生的对话：",
    ]
    for t in state.get("transcript", []):
        if t["action"] == "ask":
            lines.append(f"  {t['player']} 问：{t['content']}")
            lines.append(f"  汤主答：{t.get('host_answer', '?')}")
        else:
            lines.append(f"  {t['player']} 猜想：{t['content']}（命中率 {t.get('score', '?')}）")
    lines += [
        "",
        "请只输出一个 JSON 对象，格式：",
        '  {"action": "ask", "content": "你的 yes/no 问题"}',
        '  {"action": "guess", "content": "你对完整故事的猜想"}',
        "不要输出任何额外文字。",
    ]
    return "\n".join(lines)


def build_judge_prompt(question: str, puzzle: dict[str, Any]) -> str:
    """构造汤主（裁判）应答提示（v0.2 接入 LLM 裁判时使用）。"""
    return (
        "你是海龟汤的汤主，掌握真相。请仅根据真相用 yes/no 回答玩家问题，"
        "不得透露额外信息。\n\n"
        f"真相：{puzzle['truth']}\n\n"
        f"玩家问题：{question}\n\n"
        "只回答 yes 或 no。"
    )


def build_debrief_prompt(puzzle: dict[str, Any]) -> str:
    """构造赛后复盘提示。"""
    return (
        "请基于以下真相，对这场海龟汤对局做复盘：各方推理路径是否合理、"
        "哪些关键事实被遗漏、玩家表现如何。\n\n"
        f"真相：{puzzle['truth']}\n"
        f"关键事实：{puzzle.get('key_facts', [])}"
    )
