"""评测记录与凭证的数据模型（P1 · 链上评测凭证）。

设计要点（每一条都对应一个已踩过或可预见的坑）：

1. **确定性序列化（canonical bytes）**
   签名的对象是字节，不是 Python 对象。若序列化不确定（键序随机、浮点 repr 差异、
   中文转义方式不同），同一份记录两次序列化会得到不同字节 → 哈希不同 →
   验签必然失败。故 `canonical_bytes()` 固定：sort_keys + 紧凑分隔符 +
   ensure_ascii=False + 浮点统一 round(6)。

2. **凭证自包含**
   Credential 必须自带公钥与记录原文，否则「离线验签」无从谈起——校验方拿不到
   本项目的运行环境与密钥库。设计为：凭证 + 公开算法 = 任何人可独立验证。

3. **锚定状态与密钥保管状态分开标注**
   两个正交维度，混为一谈会让「已签名」被误读成「已上链」或「密钥安全」：
     - `anchor`：是否真的提交到了共识链（None = 仅本地签名）
     - `key_custody`：私钥如何保管（demo-insecure = 用了公开默认口令，不保密）
   两者都必须如实写入凭证，不允许静默降级而不落痕迹。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

SCHEMA_VERSION = "arena-eval-credential/v1"

# 浮点统一精度：避免 repr 差异导致同一记录算出不同哈希
_FLOAT_NDIGITS = 6


def _round_floats(obj: Any) -> Any:
    """递归把 float 收敛到固定精度（dict/list 穿透）。"""
    if isinstance(obj, float):
        return round(obj, _FLOAT_NDIGITS)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj


def canonical_bytes(payload: Dict[str, Any]) -> bytes:
    """确定性序列化：同一 dict 任何时刻、任何平台都得到相同字节。"""
    stable = _round_floats(payload)
    return json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class EvaluationRecord:
    """一局评测的客观结论——这是被签名的对象。

    刻意只放**摘要与哈希**，不放完整 transcript：
      ① transcript 体积大且含模型原文，进凭证会让凭证臃肿；
      ② 但只要 transcript_hash 被签名，原始 transcript 就同样受保护
         （改动 transcript → 哈希不符 → 验签失败），保护强度等价。
    """

    match_id: str
    puzzle_id: str
    # 参与者：{角色: 标识}，如 {"questioner": "deepseek", "host": "mock"}
    participants: Dict[str, str]
    rounds: int
    # 客观指标（口径来自 engine/evaluate.py，禁止在本模块另算一份）
    metrics: Dict[str, float]
    # 原始 transcript 的 sha256；transcript 本身不进凭证
    transcript_hash: str
    # 裁判结论：judge_id + 结论（P2-A 阶段会扩成多裁判共识列表）
    verdict: Dict[str, Any]
    created_at: str  # ISO 8601 字符串（由调用方生成，避免本模块依赖时区状态）

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "match_id": self.match_id,
            "puzzle_id": self.puzzle_id,
            "participants": self.participants,
            "rounds": self.rounds,
            "metrics": self.metrics,
            "transcript_hash": self.transcript_hash,
            "verdict": self.verdict,
            "created_at": self.created_at,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_payload())

    def digest(self) -> str:
        return sha256_hex(self.canonical_bytes())


@dataclass
class Credential:
    """评测凭证：可被任何人离线独立验证。"""

    # 被签名的原始记录（明文保留，供校验方重算哈希）
    record: Dict[str, Any]
    # 记录摘要（= 签名对象）
    record_hash: str
    # ECDSA 签名（hex）
    signature_hex: str
    # 签名者公钥（hex）——凭证自包含，校验方无需访问本项目密钥库
    public_key_hex: str
    signer_id: str
    algorithm: str  # 如 "ECDSA-P256-SHA256"
    # 密钥保管口径：见模块文档第 3 条，不允许省略
    key_custody: str
    # 链上锚定：None = 仅本地签名，未提交共识链
    anchor: Optional[Dict[str, Any]] = None
    # 后端类型，便于追溯凭证由哪条路径产出
    backend: str = "unknown"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent,
                          sort_keys=True)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Credential":
        return Credential(**d)

    @staticmethod
    def from_json(text: str) -> "Credential":
        return Credential.from_dict(json.loads(text))


# 密钥保管状态的两种取值（显式枚举，防止各处写自由字符串后口径漂移）
KEY_CUSTODY_SECURE = "passphrase-protected"   # 私钥由独立口令加密，具备保密性
KEY_CUSTODY_DEMO = "demo-insecure-default"    # 用了随源码公开的默认口令，不保密
