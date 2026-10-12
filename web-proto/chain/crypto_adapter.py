"""密码学适配层（P1）—— 口令保管、密钥对、签名/验签。

## 为什么单独一层

真实共识链（MARL-ECDSA）的 crypto 模块可直接复用，但对它存在硬路径依赖，
不能假设任何运行环境都有。本层把「用真实链的 crypto」与「用 cryptography
库自己签」统一成同一组函数，且**算法一致**（ECDSA / P-256(secp256r1) / SHA-256），
因此两条路径产出的凭证格式与验签方式完全相同，可互换、可交叉验证。

## 口令处理（本模块最关键的设计）

探测真实链时得到一条警告，必须正面处理：

> 环境变量 MARL_ECDSA_KEY_PASSPHRASE 未设置 → 回退到随源码公开的
> DEFAULT_KEY_PASSPHRASE。用它加密的私钥**不具备任何保密性**。

按本项目既有教训（静默 fallback 比报错危险），不接受「默认走不安全路径」。
三种口径：

| 情形 | 行为 | 凭证标记 |
|---|---|---|
| 已设 `MARL_ECDSA_KEY_PASSPHRASE` | 直接用，口令不落盘 | `env-passphrase` |
| 未设置（默认） | **自动生成随机口令**，存入已被 .gitignore 的本地密钥目录 | `local-generated` |
| 显式 `ARENA_ALLOW_INSECURE_KEY=1` | 允许回退公开默认口令（仅演示） | `demo-insecure-default` |

第二项是默认路径：私钥仍被真实口令加密，只是口令存在同机文件里而非仓库里。
这比「随源码公开的默认口令」强，但**弱于环境变量方式**，故如实标注区分，
不把三者混为一谈。
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Optional, Tuple

from .models import (
    KEY_CUSTODY_DEMO,
    sha256_hex,
)

KEY_CUSTODY_ENV = "env-passphrase"
KEY_CUSTODY_LOCAL = "local-generated"
# 私钥仅存内存不落盘（本地自签路径）：不存在被窃取落盘私钥的问题
KEY_CUSTODY_EPHEMERAL = "ephemeral-not-persisted"

# 真实共识链路径（可通过环境变量覆盖，避免把本机绝对路径写死在代码里）
DEFAULT_CHAIN_PATH = os.environ.get(
    "ARENA_CHAIN_PATH",
    r"E:/Program/MARL/【CCF】区块链AI协同：面向MARL的共识机制/marl-ecdsa-consensus-chain",
)

# 本地密钥目录：位于 web-proto/data/ 下，已被根 .gitignore 排除（不进仓库）
DEFAULT_KEY_DIR = Path(__file__).resolve().parents[1] / "data" / "chain_keys"

_PASSPHRASE_ENV = "MARL_ECDSA_KEY_PASSPHRASE"
_INSECURE_ENV = "ARENA_ALLOW_INSECURE_KEY"


def ensure_key_passphrase(key_dir: Path) -> Tuple[Optional[str], str]:
    """确保口令就位，返回 (口令或None, 保管口径)。

    返回 None 表示「允许回退到真实链自己的默认口令」，仅当显式开启
    ARENA_ALLOW_INSECURE_KEY=1 时发生；此时保管口径为 demo-insecure-default。
    """
    existing = os.environ.get(_PASSPHRASE_ENV)
    if existing:
        return existing, KEY_CUSTODY_ENV

    if os.environ.get(_INSECURE_ENV) == "1":
        # 显式授权的不安全路径：不设置环境变量，让下游走它自己的默认口令
        return None, KEY_CUSTODY_DEMO

    key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    pf_path = key_dir / ".passphrase"
    if pf_path.exists():
        secret = pf_path.read_text(encoding="utf-8").strip()
        if not secret:
            secret = secrets.token_hex(32)
            pf_path.write_text(secret, encoding="utf-8")
    else:
        secret = secrets.token_hex(32)
        pf_path.write_text(secret, encoding="utf-8")
        try:
            # 尽量收紧权限（Windows 上 chmod 效果有限，但 Linux/macOS 有效）
            os.chmod(pf_path, 0o600)
        except OSError:
            pass
    os.environ[_PASSPHRASE_ENV] = secret
    return secret, KEY_CUSTODY_LOCAL


# --------------------------------------------------------------------------
# 真实链 crypto 的软导入
# --------------------------------------------------------------------------

def try_import_marl_chain(path: Optional[str] = None):
    """尝试导入真实共识链的 crypto/ledger 模块；不可用返回 None。

    导入前先确保口令就位——该链在生成/加载密钥时才会解析口令，
    但提前设置可避免任何模块级解析时序问题。
    """
    p = path or DEFAULT_CHAIN_PATH
    if not p or not Path(p).exists():
        return None
    try:
        import sys
        if p not in sys.path:
            sys.path.insert(0, p)
        from blockchain.crypto.ecdsa_utils import ECDSAUtils
        from blockchain.crypto.key_manager import KeyManager
        from blockchain.ledger.blockchain import Blockchain
        from blockchain.ledger.block import Transaction
        return {
            "ECDSAUtils": ECDSAUtils,
            "KeyManager": KeyManager,
            "Blockchain": Blockchain,
            "Transaction": Transaction,
            "path": p,
        }
    except Exception:
        # 不吞掉信息但也不中断评测：由调用方降级
        return None


# --------------------------------------------------------------------------
# 密钥对与签名（两条路径，算法一致）
# --------------------------------------------------------------------------

def _local_generate_keypair():
    from cryptography.hazmat.primitives.asymmetric import ec
    return ec.generate_private_key(ec.SECP256R1())


def _local_pub_hex(priv) -> str:
    from cryptography.hazmat.primitives import serialization
    pub = priv.public_key()
    raw = pub.public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    )
    return raw.hex()


def _local_sign(priv, message: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import Prehashed

    # 注意：必须与真实链 ECDSAUtils.sign 的摘要口径一致，否则两路径签名
    # 互不兼容（凭证无法交叉验证）。真实链内部亦为 SHA-256(message) 后 ECDSA，
    # 该等价性由 tests/test_chain.py 的交叉验证用例守护。
    digest = _sha256(message)
    return priv.sign(digest, ec.ECDSA(Prehashed(hashes.SHA256())))


def _local_verify(pub_hex: str, message: bytes, signature: bytes) -> bool:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import Prehashed

    try:
        pub = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), bytes.fromhex(pub_hex)
        )
    except Exception:
        return False
    digest = _sha256(message)
    try:
        pub.verify(signature, digest, ec.ECDSA(Prehashed(hashes.SHA256())))
        return True
    except Exception:
        return False


def _sha256(data: bytes) -> bytes:
    import hashlib
    return hashlib.sha256(data).digest()


class Signer:
    """统一签名器：优先复用真实链 crypto，否则用 cryptography 直签。

    两者算法一致（ECDSA / P-256 / SHA-256），凭证可交叉验证。
    """

    def __init__(self, signer_id: str = "arena-evaluator",
                 key_dir: Optional[Path] = None,
                 chain_path: Optional[str] = None,
                 allow_insecure: bool = False):
        self.signer_id = signer_id
        self.key_dir = Path(key_dir) if key_dir else DEFAULT_KEY_DIR
        if allow_insecure:
            os.environ[_INSECURE_ENV] = "1"
        self._custody = None
        self._chain = try_import_marl_chain(chain_path)
        # 口令必须在任何密钥操作之前就位
        _, self._custody = ensure_key_passphrase(self.key_dir)
        self.using_chain_crypto = self._chain is not None
        self._init_keys()

    def _init_keys(self):
        if self._chain:
            km = self._chain["KeyManager"](key_dir=str(self.key_dir))
            self._priv, self._pub = km.generate_or_load(self.signer_id)
            self._pub_hex = self._chain["ECDSAUtils"].public_key_to_hex(self._pub)
        else:
            self.key_dir.mkdir(parents=True, exist_ok=True)
            kf = self.key_dir / f"{self.signer_id}.pubhex"
            priv = _local_generate_keypair()
            # 私钥仅存内存、不落盘 → 保管口径优于任何落盘方式，如实标注
            self._priv = priv
            self._pub_hex = _local_pub_hex(priv)
            self._custody = KEY_CUSTODY_EPHEMERAL
            try:
                kf.write_text(self._pub_hex, encoding="utf-8")
            except OSError:
                pass

    @property
    def public_key_hex(self) -> str:
        return self._pub_hex

    @property
    def key_custody(self) -> str:
        return self._custody or KEY_CUSTODY_LOCAL

    @property
    def algorithm(self) -> str:
        return "ECDSA-P256-SHA256"

    def sign(self, message: bytes) -> bytes:
        if self._chain:
            return self._chain["ECDSAUtils"].sign(self._priv, message)
        return _local_sign(self._priv, message)

    def sign_hex(self, message: bytes) -> str:
        return self.sign(message).hex()


def verify_signature(public_key_hex: str, message: bytes,
                     signature_hex: str,
                     chain_path: Optional[str] = None) -> bool:
    """验签。优先用真实链的 verify，否则用本地实现。

    两者对同一 (公钥, 消息, 签名) 必须给出相同结论——这是「算法一致」的
    可执行保证，已由 tests/test_chain.py 交叉验证守护。
    """
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False

    chain = try_import_marl_chain(chain_path)
    if chain:
        try:
            pub = chain["ECDSAUtils"].public_key_from_hex(public_key_hex)
            return bool(chain["ECDSAUtils"].verify(pub, message, signature))
        except Exception:
            # 真实链验签异常时回退本地实现，不静默判为失败
            pass
    return _local_verify(public_key_hex, message, signature)
