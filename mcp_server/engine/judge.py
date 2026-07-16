"""裁判逻辑（v0.1 规则版，不依赖 LLM）。

说明：v0.1 用 Mock 玩家离线跑通，裁判用"关键词重叠"启发式判定 yes/no 与猜想命中率。
v0.2 接入真实模型后，可换成 LLM 裁判（结合 prompts.judge_prompt）。
"""
from __future__ import annotations

import re
from typing import Any

# 中文常见停用字（用于去噪，提升重叠比可信度）
_STOP = set("的了在是和与及对也都被把让给从到把而而且并且但但是一个这名位他她它我们你们他们这那其上中下里外前后左右没有不没无否")


def _shingles(text: str) -> set[str]:
    """把文本切成 2 字片段集合（中文友好）。"""
    text = re.sub(r"\s+", "", text)
    if len(text) <= 1:
        return {text}
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _overlap(a: str, b: str) -> float:
    """Jaccard 重叠度。"""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def answer_question(question: str, puzzle: dict[str, Any]) -> tuple[str, str | None]:
    """根据 truth / key_facts 回答 yes/no，并返回命中的关键事实（用于向玩家揭示）。

    规则：取与问题重叠度最高的关键事实；重叠度 ≥ 阈值判为 yes 并揭示该事实，否则 no。
    """
    facts: list[str] = puzzle.get("key_facts", [])
    best_fact, best_score = None, 0.0
    for f in facts:
        s = _overlap(question, f)
        if s > best_score:
            best_score, best_fact = s, f
    if best_score >= 0.18:
        return "yes", best_fact
    return "no", None


def evaluate_guess(guess: str, puzzle: dict[str, Any], threshold: float) -> tuple[bool, float]:
    """评估猜想：关键事实命中率 ≥ threshold 即获胜。

    命中 = 该关键事实与猜想的重叠度 ≥ 阈值。返回 (是否获胜, 命中率)。
    """
    facts: list[str] = puzzle.get("key_facts", [])
    if not facts:
        return False, 0.0
    hit = sum(1 for f in facts if _overlap(guess, f) >= 0.18)
    score = hit / len(facts)
    return score >= threshold, score


def debrief_text(puzzle: dict[str, Any]) -> str:
    """生成赛后复盘 / 真相揭示文本。"""
    return (
        f"【题目】{puzzle['title']}\n"
        f"【汤面】{puzzle['surface']}\n\n"
        f"【汤底（真相）】\n{puzzle['truth']}\n\n"
        f"【关键事实】\n" + "\n".join(f"- {f}" for f in puzzle.get("key_facts", []))
    )
