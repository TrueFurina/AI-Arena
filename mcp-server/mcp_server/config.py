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

# 离线玩家（无需任何 API key，用于本地跑通与对照评测）
# - mock：预言机，知道真相，仅用于验证流程，**不参与能力评测**
# - heuristic*：不作弊的启发式玩家，行为差异带来真实的得分区分度
OFFLINE_PLAYERS: dict[str, str] = {
    "mock": "预言机玩家：掌握真相，仅用于跑通流程，不参与能力评测",
    "heuristic": "启发式玩家（均衡）：汤面词 + 常识假设交替提问，六成进度后收网",
    "heuristic-aggressive": "启发式玩家（激进）：先赌大方向、提问少、收网早",
    "heuristic-conservative": "启发式玩家（保守）：穷举汤面细节、提问多、收网晚",
    "heuristic-creative": "启发式玩家（跳跃）：大量通用假设、顺序打乱、方差大",
    # deception 模式：知情者不是求解者，不参与求解者排名，只评估欺骗/伪装能力
    "insider": "知情者（摇摆策略）：真话假话混着说，最难对付",
    "insider-honest": "知情者（诚实）：始终如实回答，用作对照组基准",
    "insider-misleading": "知情者（激进欺骗）：始终反向回答，规律明显易被识破",
    "insider-mixed": "知情者（摇摆）：50% 概率撒谎",
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
    """返回模型注册表快照（不含任何 key），供 arena://models 资源使用。

    除真实 LLM 外，还列出离线玩家（mock / heuristic*），便于客户端直接开本地对照局。
    """
    out = []
    for name, (model_str, env) in MODEL_REGISTRY.items():
        out.append(
            {
                "name": name,
                "litellm_model": model_str,
                "configured": has_api_key(name),
                "offline": False,
                "description": "真实模型（需配置 API key）",
            }
        )
    for name, desc in OFFLINE_PLAYERS.items():
        out.append(
            {
                "name": name,
                "litellm_model": None,
                "configured": True,  # 离线玩家无需 key，始终可用
                "offline": True,
                "description": desc,
            }
        )
    return out
