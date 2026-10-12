"""玩家接口定义。

- Player：提问者（questioner），负责问问题与终局还原。
- Host：出题人/汤主（answerer），知道汤底，回答 yes/no 类问题。
- Judge：终局打分器。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Player(ABC):
    def __init__(self, id: str, model: str):
        self.id = id
        self.model = model

    @abstractmethod
    def ask_question(self, puzzle, transcript: List[dict]) -> str:
        """根据当前对局记录，提出下一个 yes/no 问题。"""

    @abstractmethod
    def final_guess(self, puzzle, transcript: List[dict]) -> str:
        """终局：写出对汤底的还原（一段文字）。"""


class Host(ABC):
    def __init__(self, id: str = "host", model: str = "host"):
        self.id = id
        self.model = model

    @abstractmethod
    def answer(self, puzzle, question: str, transcript: List[dict]) -> str:
        """回答提问者的 yes/no 问题（是 / 否 / 部分是 / 无关 / 不重要）。"""

    def reset(self, puzzle) -> None:
        """每局开始前由引擎调用（默认 no-op）。

        有每局状态的主持（如 P2-B DeceptiveHost 的谎言配额）覆写本方法。
        TurtleSoupGame.run 与 players.rl.run_episode 都以 hasattr 守卫
        调用——第三方主持不继承本 ABC 时也不会炸，行为零变化。
        """


class Judge(ABC):
    @abstractmethod
    def score(self, puzzle, guess: str):
        """返回 (score: float 0-100, rationale: str)。"""
        raise NotImplementedError
