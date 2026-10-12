"""players 包：提问者 / 主持 / 裁判。"""
from .base import Player, Host, Judge
from .mock import MockQuestioner, MockHost, MockJudge
from .llm import (
    LLMQuestioner, LLMHost, LLMJudge,
    env_key_for_model, make_player,
)

__all__ = [
    "Player", "Host", "Judge",
    "MockQuestioner", "MockHost", "MockJudge",
    "LLMQuestioner", "LLMHost", "LLMJudge",
    "env_key_for_model", "make_player",
]
