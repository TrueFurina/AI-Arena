"""离线验签（P1）—— 任何人拿到凭证即可独立验证，无需访问本项目运行环境。

校验三步，任一步失败即判为不可信：
  ① 重算记录摘要：与凭证里的 record_hash 比对（防记录被改）
  ② ECDSA 验签：用凭证自带公钥验证签名（防伪造签发者）
  ③ 锚定状态说明：若 anchor 存在则附带其 status（in-pool 不具最终性）

刻意不做的两件事：
  - 不访问任何密钥库或配置文件：凭证自包含公钥，校验方零依赖
  - 不把「入池」当「已锚定」：status 原样透出，由调用方判断
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .crypto_adapter import verify_signature
from .models import canonical_bytes, sha256_hex, Credential


@dataclass
class VerificationResult:
    ok: bool
    steps: Dict[str, bool]
    reasons: Dict[str, str]
    anchor_status: Optional[str] = None
    key_custody: Optional[str] = None

    def summary(self) -> str:
        flag = "✅ 可信" if self.ok else "❌ 不可信"
        detail = "；".join(f"{k}={'通过' if v else '失败'}"
                          for k, v in self.steps.items())
        extra = ""
        if self.key_custody:
            extra += f"；密钥保管={self.key_custody}"
        if self.anchor_status:
            extra += f"；锚定={self.anchor_status}"
        reasons = ""
        if self.reasons:
            reasons = "（" + "；".join(self.reasons.values()) + "）"
        return f"{flag}：{detail}{extra}{reasons}"


def verify_credential(credential: Credential,
                      chain_path: Optional[str] = None) -> VerificationResult:
    steps: Dict[str, bool] = {}
    reasons: Dict[str, str] = {}

    # ① 记录摘要一致性
    try:
        recomputed = sha256_hex(canonical_bytes(credential.record))
        steps["digest"] = (recomputed == credential.record_hash)
        if not steps["digest"]:
            reasons["digest"] = (f"记录摘要不符（重算 {recomputed[:12]}… ≠ "
                                 f"凭证 {credential.record_hash[:12]}…）")
    except Exception as exc:
        steps["digest"] = False
        reasons["digest"] = f"摘要重算异常 {type(exc).__name__}"

    # ② 签名验证（签的是 canonical_bytes(record)，不是摘要字符串）
    message = canonical_bytes(credential.record)
    try:
        steps["signature"] = verify_signature(
            credential.public_key_hex, message,
            credential.signature_hex, chain_path=chain_path,
        )
        if not steps["signature"]:
            reasons["signature"] = "签名校验失败（记录被篡改或签名不匹配公钥）"
    except Exception as exc:
        steps["signature"] = False
        reasons["signature"] = f"验签异常 {type(exc).__name__}"

    ok = all(steps.values())
    anchor_status = None
    if credential.anchor:
        anchor_status = credential.anchor.get("status")

    return VerificationResult(
        ok=ok,
        steps=steps,
        reasons=reasons,
        anchor_status=anchor_status,
        key_custody=credential.key_custody,
    )


def verify_credential_json(text: str,
                           chain_path: Optional[str] = None) -> VerificationResult:
    return verify_credential(Credential.from_json(text), chain_path=chain_path)
