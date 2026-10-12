"""离线 Mock 玩家：无需任何网络/API 即可跑通完整流程，用于先验证引擎。"""
from __future__ import annotations

from typing import List

from .base import Player, Host, Judge
from engine.game import Puzzle

# 通用问题池（覆盖常见推理方向），mock 提问者从中轮流取用
_QUESTION_BANK = [
    "这件事和死有关吗？",
    "死者是自杀的吗？",
    "现场有其他人吗？",
    "这件事和钱有关吗？",
    "凶手是熟人吗？",
    "死者之前经历过什么创伤吗？",
    "这件事发生在室外吗？",
    "有交通工具参与吗？",
    "死者认识凶手吗？",
    "汤底里出现了'吃'的行为吗？",
    "有亲属关系牵连吗？",
    "真相和'误解'有关吗？",
]


class MockQuestioner(Player):
    def __init__(self, id: str, model: str = "mock-gpt4o", seed: int = 0):
        super().__init__(id, model)
        self.seed = seed

    def ask_question(self, puzzle: Puzzle, transcript: List[dict]) -> str:
        # 依据历史轮数从问题池里取，保证不同提问者/轮次有差异
        n = len([m for m in transcript if m["role"] == "question"])
        q = _QUESTION_BANK[(n + self.seed) % len(_QUESTION_BANK)]
        return q

    def final_guess(self, puzzle: Puzzle, transcript: List[dict]) -> str:
        return (
            f"【{self.id} 的还原】根据主持的回答，我推断："
            f"此事与死亡和人际关系有关，存在某种关键误解；"
            f"汤面『{puzzle.surface}』的真相应涉及隐藏的身份或情感动机。"
        )


class MockHost(Host):
    def __init__(self):
        super().__init__(id="host", model="mock-host")

    def answer(self, puzzle: Puzzle, question: str, transcript: List[dict]) -> str:
        for qa in puzzle.golden_qa:
            if qa.get("contains") and qa["contains"] in question:
                return qa["answer"]
        return "无关"


def _char_f1(a: str, b: str) -> float:
    """字符级 F1，作为无嵌入时的粗略相似度。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    if not inter:
        return 0.0
    prec = len(inter) / len(sa)
    rec = len(inter) / len(sb)
    return 2 * prec * rec / (prec + rec)


class MockJudge(Judge):
    def __init__(self, model: str = "mock-judge"):
        self.model = model

    def score(self, puzzle: Puzzle, guess: str):
        s = _char_f1(guess, puzzle.solution) * 100
        rationale = f"字符级重叠相似度 ≈ {s:.1f}%（mock 打分，仅用于流程验证）"
        return round(s, 1), rationale
