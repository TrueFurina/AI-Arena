"""链上评测凭证（P1 · L3 链上可信的最小可用层）+ 身份锚定（P2-A）。

对外接口：
    from chain.recorder import MatchRecorder
    from chain.verify import verify_credential
    from chain.identity import IdentityRegistry

    rec = MatchRecorder()
    cred = rec.issue_from_result(result_dict)
    print(verify_credential(cred).summary())

    # 可选：接入身份锚定（挡住「自造密钥冒充已登记裁判」）
    reg = IdentityRegistry()
    print(verify_credential(cred, identity=reg).summary())

设计原则见各模块文档。三条最要紧的：
  1. 指标口径只来自 engine/evaluate.py，本层不计算指标
  2. 链不可用 → 降级为仅本地签名，anchor=None，绝不阻塞评测；
     降级状态写入凭证的 anchor/key_custody 字段，不允许静默发生

## ⚠️ 能力边界（必须随凭证一起声明，不得省略）

**完整性（integrity）**：验签通过只证明「凭证内的记录自签发以来未被
篡改」。凭证自带公钥意味着：**不接身份锚定时，任何人都能自造一对密钥、
签发一份看起来完全合法的假凭证，且 verify_credential 会通过**。

**身份可信（authenticity）**：P2-A 已补上这一半——`chain/identity.py`
把 agent_id → 公钥的绑定交给 MARL-ECDSA 的 identity_contract，验签时
比对凭证公钥与合约登记是否一致（`verify_credential(cred, identity=reg)`）。
冒名凭证在该步被拦下（mismatch）。但它**不是全局 PKI**：
  · 身份合约 WorldState 是进程内内存态，本层不做持久化接线——
    「谁有资格注册」这个问题本层没有回答，信任根来源属部署问题
  · 注册先到先得：挡得住「冒名顶替换公钥」，挡不住「抢占没人认领的 id」，
    故只应登记已知且受控的裁判
  · 合约不可用 → identity_status=unavailable，该步不参与 ok 判定
    （既不判通过也不据此判不可信），如实标注「未做身份校验」

也就是说：**验签通过 ≠ 这份凭证是本评测台签发的**，除非同时接入了
身份锚定且该 agent_id 已受控登记。报告与对外叙述中，未接身份锚定的
凭证不得表述为「已防伪」或「来源可认证」。
"""

from .models import Credential, EvaluationRecord  # noqa: F401
from .crypto_adapter import Signer, verify_signature  # noqa: F401
from .recorder import MatchRecorder  # noqa: F401
from .verify import verify_credential, VerificationResult  # noqa: F401
from .identity import IdentityRegistry  # noqa: F401

__all__ = [
    "Credential",
    "EvaluationRecord",
    "Signer",
    "MatchRecorder",
    "verify_credential",
    "VerificationResult",
    "verify_signature",
    "IdentityRegistry",
]
