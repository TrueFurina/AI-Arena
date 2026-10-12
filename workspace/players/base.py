"""玩家抽象接口。对应 specs/engine-design.md §4/§5。"""

from abc import ABC, abstractmethod


class Player(ABC):
    """提问者基类：输入 observation，输出动作（ask / guess）。"""

    name = "player"

    @abstractmethod
    def act(self, observation: dict) -> dict:
        """返回 {'action': 'ask'|'guess', 'content': str}。"""
        raise NotImplementedError


class GM(ABC):
    """主持人基类：输入问题，输出是非回答。"""

    @abstractmethod
    def answer(self, question: str, context: dict) -> dict:
        """返回 {'answer': 'yes'|'no'|...}。context 含真相与历史（不泄漏给提问者）。"""
        raise NotImplementedError
