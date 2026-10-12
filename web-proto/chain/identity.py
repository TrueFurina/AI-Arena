"""签发者身份锚定（补 P1 缺口）——让「谁签的」可被校验，而不只是「没被改」。

## P1 留下的缺口

`chain/verify.py` 能证明**记录自签名后未被篡改**，但凭证自带公钥，因此
**任何人都能自造一对密钥、签发一份自洽的假凭证并验签通过**。当时已把该
边界写进文档并固化为一条测试断言（伪造凭证本就应验签通过）。

本模块补上另一半：把 `agent_id → 公钥` 的绑定交给身份合约托管，
校验时比对「凭证里的公钥」与「合约里该 agent_id 登记的公钥」是否一致。
两者齐备，伪造者就无法再拿自造密钥冒充已知裁判。

## ⚠️ 三条必须说清的边界

  ① **这不是全局 PKI**。身份合约的 WorldState 是进程内内存态，本模块不做
     持久化接线。因此「谁有资格往里注册」这个问题，本模块**没有回答**——
     它只提供绑定与比对机制。信任根的来源属部署问题，不属于本层能力。
  ② **注册先到先得**：实测身份合约拒绝对已注册 agent_id 重复注册
     （返回 success=False），这挡住了「冒名顶替换公钥」，但挡不住
     「抢先注册一个没人认领的 id」。故本层只应登记**已知且受控**的裁判。
  ③ **注册失败必须显式返回，不得静默**：实测非法公钥格式会返回
     `{'success': False, 'error': ...}`。本模块原样上抛该失败，
     不把「注册没成」伪装成「已锚定」。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from chain.crypto_adapter import DEFAULT_CHAIN_PATH, Signer

# 身份校验的四种结果（显式枚举，防止各处写自由字符串后口径漂移）
IDENTITY_OK = "ok"                        # 公钥与合约登记一致
IDENTITY_MISMATCH = "mismatch"            # 公钥与登记不一致（冒名）
IDENTITY_UNREGISTERED = "unregistered"    # 该 agent_id 未登记
IDENTITY_UNAVAILABLE = "unavailable"      # 身份合约不可用，未做校验


def try_import_identity(chain_path: Optional[str] = None):
    """尝试导入真实身份合约；不可用返回 None。"""
    p = chain_path or DEFAULT_CHAIN_PATH
    if not p:
        return None
    try:
        import sys
        from pathlib import Path
        if not Path(p).exists():
            return None
        if p not in sys.path:
            sys.path.insert(0, p)
        from blockchain.contracts.identity_contract import IdentityContract
        from blockchain.ledger.world_state import WorldState
        return {"IdentityContract": IdentityContract, "WorldState": WorldState,
                "path": p}
    except Exception:
        return None


class IdentityRegistry:
    """`agent_id → 公钥` 的绑定与校验。

    链不可用时不抛异常，而是让 `status` 恒为 `IDENTITY_UNAVAILABLE`——
    调用方据此如实标注「未做身份校验」，而不是让它看起来通过了。
    """

    def __init__(self, chain_path: Optional[str] = None):
        self.chain_path = chain_path
        self._mod = try_import_identity(chain_path)
        self._contract = None
        if self._mod is not None:
            try:
                self._contract = self._mod["IdentityContract"](
                    self._mod["WorldState"]())
            except Exception:
                self._contract = None

    @property
    def available(self) -> bool:
        return self._contract is not None

    def register(self, agent_id: str, public_key_hex: str) -> Dict[str, Any]:
        """登记绑定。失败（含重复注册/格式非法）原样返回，不静默。"""
        if not self.available:
            return {"success": False, "error": IDENTITY_UNAVAILABLE,
                    "reason": "身份合约不可用，未登记"}
        try:
            return self._contract.register(agent_id, public_key_hex)
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    def register_signer(self, signer: Signer,
                        agent_id: Optional[str] = None) -> Dict[str, Any]:
        return self.register(agent_id or signer.signer_id,
                             signer.public_key_hex)

    def check(self, agent_id: str, public_key_hex: str) -> Tuple[str, str]:
        """校验 (agent_id, 公钥) 是否与合约登记一致。

        返回 (status, message)。status 取值见模块顶部四枚举。
        """
        if not self.available:
            return (IDENTITY_UNAVAILABLE,
                    "身份合约不可用 → 未做签发者身份校验（仅完整性已验证）")
        try:
            known = self._contract.get_public_key(agent_id)
        except Exception as e:
            return (IDENTITY_UNAVAILABLE,
                    f"身份合约查询失败：{type(e).__name__}: {e}")
        if known is None:
            return (IDENTITY_UNREGISTERED,
                    f"agent_id '{agent_id}' 未在身份合约登记")
        if known != public_key_hex:
            return (IDENTITY_MISMATCH,
                    f"公钥不匹配：凭证中的公钥与合约登记的 '{agent_id}' 不一致"
                    " → 可能为冒名签发")
        if not self._contract.is_active(agent_id):
            return (IDENTITY_MISMATCH,
                    f"agent_id '{agent_id}' 在身份合约中非 active 状态")
        return (IDENTITY_OK, f"'{agent_id}' 身份校验通过")
