"""配置加载：.env 读取与模型注册表。

关键约定：
- 所有敏感 key 仅从此模块读取，绝不写死在业务代码里。
- 模型以"友好名"注册，映射到 litellm 的模型串；调用层无感知。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（mcp_server 的上一级）
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "mcp_server" / "data"
MATCHES_DIR = DATA_DIR / "matches"

# 加载 .env（若存在）
load_dotenv(ROOT / ".env")


def _getenv(name: str, default: str = "") -> str:
    return os.getenv(name, default)


# 友好名 -> (litellm 模型串, 对应 API key 环境变量)
MODEL_REGISTRY: dict[str, tuple[str, str | None]] = {
    "gpt-4o-mini": ("gpt-4o-mini", "OPENAI_API_KEY"),
    "gpt-4o": ("gpt-4o", "OPENAI_API_KEY"),
    "claude-3.5-sonnet": ("anthropic/claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY"),
    "deepseek-chat": ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY"),
    "qwen-plus": ("openai/qwen-plus", "DASHSCOPE_API_KEY"),   # 通义，走兼容端点
    "kimi": ("moonshot/moonshot-v1-8k", "MOONSHOT_API_KEY"),
}

# 默认参数
DEFAULT_MAX_ROUNDS = int(_getenv("ARENA_DEFAULT_MAX_ROUNDS", "20"))
GUESS_THRESHOLD = float(_getenv("ARENA_GUESS_THRESHOLD", "0.6"))


def resolve_model(friendly: str) -> tuple[str, str | None]:
    """把友好名解析成 (litellm 模型串, key 环境变量名)。

    未知名称原样返回（litellm 也支持直接传模型串），key 环境变量置 None。
    """
    return MODEL_REGISTRY.get(friendly, (friendly, None))


def has_api_key(friendly: str) -> bool:
    """该模型是否已配置可用 key。"""
    _, env = resolve_model(friendly)
    if env is None:
        return False
    return bool(_getenv(env))


def list_models() -> list[dict]:
    """返回模型注册表快照（不含任何 key），供 arena://models 资源使用。"""
    out = []
    for name, (model_str, env) in MODEL_REGISTRY.items():
        out.append(
            {
                "name": name,
                "litellm_model": model_str,
                "configured": has_api_key(name),
            }
        )
    return out
