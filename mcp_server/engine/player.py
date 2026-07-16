"""玩家适配层：统一接口调多种模型（litellm），并以 Mock 兜底。

设计要点：
- BasePlayer.act(state) -> {"action": "ask"|"guess", "content": str}
- MockPlayer：知道真相的"预言机"玩家，仅用于离线跑通流程，不用于真实评测。
- LLMPlayer：通过 litellm 调真实模型，解析其 JSON 动作；调用失败时安全降级为一次保守提问。
- get_player(model, puzzle)：工厂。model 以 'mock' 开头或无 key → MockPlayer，否则 LLMPlayer。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mcp_server.config import GUESS_THRESHOLD, has_api_key, resolve_model
from mcp_server.engine.prompts import build_player_prompt

logger = logging.getLogger("ai-arena.player")

try:
    import litellm  # 可选依赖
    _LITELLM_AVAILABLE = True
except Exception:  # noqa: BLE001
    _LITELLM_AVAILABLE = False


class BasePlayer:
    def __init__(self, name: str, puzzle: dict[str, Any]):
        self.name = name
        self.puzzle = puzzle

    def act(self, state: dict[str, Any]) -> dict[str, str]:
        raise NotImplementedError


class MockPlayer(BasePlayer):
    """预言机玩家：依次就每个关键事实提问（让汤主揭示），最后用已知事实完美重构猜想。

    仅用于验证端到端流程，因其"作弊"式掌握真相，不应用于能力评测。
    """

    def __init__(self, name: str, puzzle: dict[str, Any]):
        super().__init__(name, puzzle)
        self.facts: list[str] = puzzle.get("key_facts", [])
        self._i = 0

    def act(self, state: dict[str, Any]) -> dict[str, str]:
        if self._i < len(self.facts):
            fact = self.facts[self._i]
            self._i += 1
            return {"action": "ask", "content": f"这件事是不是和「{fact}」有关？"}
        # 已问完所有事实，提交完美重构
        return {"action": "guess", "content": "。".join(self.facts)}


class LLMPlayer(BasePlayer):
    """真实模型玩家：用 litellm 统一接口，解析 JSON 动作。"""

    def __init__(self, name: str, puzzle: dict[str, Any]):
        super().__init__(name, puzzle)
        if not _LITELLM_AVAILABLE:
            raise RuntimeError("litellm 未安装，无法使用 LLMPlayer（pip install 'ai-arena[llm]'）")
        self.model_str, self._env = resolve_model(name)
        if not has_api_key(name):
            raise RuntimeError(f"模型 {name} 未配置 API key（检查 .env 的 {self._env}）")

    def act(self, state: dict[str, Any]) -> dict[str, str]:
        prompt = build_player_prompt(state, self.name)
        try:
            resp = litellm.completion(
                model=self.model_str,
                messages=[
                    {"role": "system", "content": "你是海龟汤玩家，严格按 JSON 格式输出。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            action = data.get("action")
            text = data.get("content", "")
            if action in ("ask", "guess") and text:
                return {"action": action, "content": text}
        except Exception as e:  # noqa: BLE001
            logger.warning("LLMPlayer %s 调用失败，降级为保守提问: %s", self.name, e)
        # 安全降级：返回一次保守提问，保证对局不中断
        return {"action": "ask", "content": "你能再给一点线索吗？"}


def get_player(model: str, puzzle: dict[str, Any]) -> BasePlayer:
    """玩家工厂：mock 或无 key 走 MockPlayer，否则 LLMPlayer（失败自动降级为 Mock-friendly 行为）。"""
    if model.lower().startswith("mock") or not has_api_key(model):
        return MockPlayer(model, puzzle)
    try:
        return LLMPlayer(model, puzzle)
    except Exception as e:  # noqa: BLE001
        logger.warning("无法创建 LLMPlayer(%s)，改用 MockPlayer: %s", model, e)
        return MockPlayer(model, puzzle)


# 兼容未使用变量引用（避免 linter 误报）
_ = GUESS_THRESHOLD
