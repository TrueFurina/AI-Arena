"""规则化假玩家，零 key 跑通 pipeline。对应 specs §1 mock 模式。

注意：这是演示用 mock——提问者被预置了关键事实用于生成问题，GM 基于真相回答。
真实模型玩家见 players/llm_player.py（M1 接入 litellm）。
"""

import re

from .base import GM, Player


def _to_question(fact: str) -> str:
    """把关键事实粗略转成是非问句。"""
    fact = fact.strip()
    fact = re.sub(r"[。！？\.！\?]+$", "", fact)
    return f"{fact}吗？"


class MockQuestioner(Player):
    name = "mock"

    def __init__(self, key_facts=None, bottom="", max_questions=5):
        self.key_facts = key_facts or []
        self.bottom = bottom
        self.max_questions = max_questions
        self._asked = 0

    def act(self, observation: dict) -> dict:
        if self._asked < len(self.key_facts) and self._asked < self.max_questions:
            fact = self.key_facts[self._asked]
            self._asked += 1
            return {"action": "ask", "content": _to_question(fact)}
        # 问完关键事实后，声明汤底
        return {"action": "guess", "content": self.bottom}


class MockGM(GM):
    name = "mock"

    def answer(self, question: str, context: dict) -> dict:
        key_facts = context.get("key_facts", [])
        for fact in key_facts:
            if fact and fact in question:
                return {"answer": "yes"}
        for clue in context.get("clue_gradient", []):
            if clue and clue in question:
                return {"answer": "yes_and_no"}
        return {"answer": "irrelevant"}


class WeakMockQuestioner(Player):
    """弱提问者基线：只问与真相**无关**的问题，用于对照（还原度应≈0）。

    配合 SmartGM 可演示「强弱玩家还原度 1.00 vs 0.00」的区分度，
    是零 key 下验证指标有效性的最小基线。对应 tests/test_m2_smoke.py 的弱玩家。
    """

    name = "mock-weak"
    _IRRELEVANT = ["今天天气好吗？", "你喜欢猫还是狗？", "现在几点了？", "你吃过饭了吗？"]

    def __init__(self, key_facts=None, bottom="", max_questions=5):
        self.bottom = bottom
        self.max_questions = max_questions
        self._asked = 0

    def act(self, observation: dict) -> dict:
        if self._asked < len(self._IRRELEVANT) and self._asked < self.max_questions:
            q = self._IRRELEVANT[self._asked]
            self._asked += 1
            return {"action": "ask", "content": q}
        return {"action": "guess", "content": self.bottom}
