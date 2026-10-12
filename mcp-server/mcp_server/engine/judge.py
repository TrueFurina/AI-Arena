"""裁判逻辑（v0.2 规则版，不依赖 LLM）。

判定核心：用「覆盖率」衡量一个问题/猜想是否触及某个关键事实。

为什么从 Jaccard 换成覆盖率：
    Jaccard 会把问题里的模板词（"……是关键吗"）计入并集，问题越长、重叠度被稀释得
    越厉害，导致"明明问到了点子上"的长问题被误判为 no。而海龟汤判定的真实语义是
    **"这个问题是否触及该事实"** —— 应当看事实被问题覆盖了多少，而非两者整体相似度。

    例：问题「地铁检修是关键吗？」 vs 事实「他是地铁检修工」
    - Jaccard  ≈ 0.10（模板词稀释）→ 判 no ✗
    - 覆盖率 = 3/6 = 0.50（"地铁/铁检/检修"三片段命中）→ 判 yes ✓

v0.2 启发式玩家靠汤面词汇 + 常识假设提问，覆盖率判定让它能真实命中事实，
从而产生有意义的得分区分度。v0.3 可替换为 LLM 裁判（见 engine/prompts.py）。
"""
from __future__ import annotations

import re
from typing import Any

# 中文常见停用字（用于去噪，提升判定可信度）
_STOP = set("的了在是和与及对也都被把让给从到把而而且并且但但是一个这名位他她它我们你们他们这那其上中下里外前后左右没有不没无否")

# 判定阈值：事实的 2-gram 有多大比例出现在问题/猜想中，即算"触及"
RELEVANCE_THRESHOLD = 0.15
# 强证据阈值：问题/猜想中包含事实里连续这么多个字，直接判为命中（无需算比例）。
# 用 2 字而非 3 字：覆盖率会随事实长度衰减（长事实永远够不到阈值），而"连续 2 字命中"
# 与事实长短无关，对短提问更公平——玩家问出汤面里的词（地铁/列车长）就该得到反馈。
LCS_THRESHOLD = 2


def _shingles(text: str) -> set[str]:
    """把文本切成 2 字片段集合（中文友好）。"""
    text = re.sub(r"\s+", "", text)
    if len(text) <= 1:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _overlap(a: str, b: str) -> float:
    """Jaccard 重叠度（保留用于兼容与调试，判定已改用 _relevance）。"""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _relevance(text: str, fact: str) -> float:
    """text 对 fact 的覆盖率：fact 的 2-gram 有多大比例出现在 text 中。

    语义：text（提问或猜想）是否"说到了"fact 的关键内容。
    取值 0~1；1 表示 text 完整包含 fact。
    """
    st, sf = _shingles(text), _shingles(fact)
    if not sf:
        return 0.0
    return len(st & sf) / len(sf)


def _longest_common_len(text: str, fact: str) -> int:
    """fact 中最长连续出现在 text 里的子串长度（从长到短穷举，fact 很短故开销可接受）。

    只统计**含实词**的公共子串：像「他是」「的了」这类纯虚词巧合不算命中，
    否则任何带"他是"的提问都能命中"他是地铁检修工"。
    """
    t = re.sub(r"\s+", "", text)
    f = re.sub(r"\s+", "", fact)
    if not t or not f:
        return 0
    for size in range(len(f), 0, -1):
        for i in range(len(f) - size + 1):
            sub = f[i : i + size]
            if sub in t and any(ch not in _STOP for ch in sub):
                return size
    return 0


def hits_fact(text: str, fact: str) -> bool:
    """text（提问或猜想）是否触及 fact。

    命中条件二选一：
    1. 强证据：text 包含 fact 中连续 ≥ LCS_THRESHOLD 个字；
    2. 弱证据：fact 的 2-gram 被 text 覆盖比例 ≥ RELEVANCE_THRESHOLD。
    """
    if _longest_common_len(text, fact) >= LCS_THRESHOLD:
        return True
    return _relevance(text, fact) >= RELEVANCE_THRESHOLD


def answer_question(question: str, puzzle: dict[str, Any]) -> tuple[str, str | None]:
    """根据 truth / key_facts 回答 yes/no，并返回命中的关键事实（用于向玩家揭示）。

    规则：取被问题覆盖率最高的关键事实；若命中则判 yes 并揭示该事实，否则 no。
    """
    facts: list[str] = puzzle.get("key_facts", [])
    best_fact, best_score = None, 0.0
    for f in facts:
        s = _relevance(question, f)
        if s > best_score:
            best_score, best_fact = s, f
    if best_fact is not None and hits_fact(question, best_fact):
        return "yes", best_fact
    return "no", None


def evaluate_guess(guess: str, puzzle: dict[str, Any], threshold: float) -> tuple[bool, float]:
    """评估猜想：关键事实命中率 ≥ threshold 即获胜。

    命中判定与提问一致（见 hits_fact）。返回 (是否获胜, 命中率)。
    """
    facts: list[str] = puzzle.get("key_facts", [])
    if not facts:
        return False, 0.0
    hit = sum(1 for f in facts if hits_fact(guess, f))
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
