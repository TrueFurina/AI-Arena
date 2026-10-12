"""链上评测凭证（P1 · L3 链上可信的最小可用层）。

对外接口：
    from chain.recorder import MatchRecorder
    from chain.verify import verify_credential

    rec = MatchRecorder()
    cred = rec.issue_from_result(result_dict)
    print(verify_credential(cred).summary())

设计原则见各模块文档。三条最要紧的：
  1. 指标口径只来自 engine/evaluate.py，本层不计算指标
  2. 链不可用 → 降级为仅本地签名，anchor=None，绝不阻塞评测；
     降级状态写入凭证的 anchor/key_custody 字段，不允许静默发生

## ⚠️ 能力边界（必须随凭证一起声明，不得省略）

本层提供的是**完整性**（integrity），**不是签发者身份可信**（authenticity）：

    凭证自带公钥 → 任何人都能自造一对密钥、签发一份看起来完全合法的
    假凭证，且 verify_credential 会通过（摘要自洽、签名自洽）。

也就是说：**验签通过 ≠ 这份凭证是本评测台签发的**。验签只能证明
「凭证内的记录自签发以来未被篡改」。

要补上身份可信，公钥必须被信任根锚定——这正是 MARL-ECDSA 的
`identity_contract` 所做的事，列为 P2 阶段工作（身份注册 + 验签时校验
公钥是否已登记）。在此之前，凭证的可信度等同于「你信任手上的这份公钥」，
报告与对外叙述中不得表述为「已防伪」或「来源可认证」。
"""

from .models import Credential, EvaluationRecord  # noqa: F401
from .crypto_adapter import Signer, verify_signature  # noqa: F401
from .recorder import MatchRecorder  # noqa: F401
from .verify import verify_credential, VerificationResult  # noqa: F401

__all__ = [
    "Credential",
    "EvaluationRecord",
    "Signer",
    "MatchRecorder",
    "verify_credential",
    "VerificationResult",
    "verify_signature",
]
