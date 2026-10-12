"""签发者身份锚定（补 P1 缺口）的协议守护。

P1 的已知边界：凭证自带公钥 ⇒ 任何人可自造密钥对签发自洽假凭证且**验签
通过**（该边界当时已固化为 test_chain.py 里的一条断言）。
本套件验证：接入身份锚定后，冒名凭证会被拦住。

⚠️ 身份合约是**进程内内存态**（WorldState() 起步为空），本层不做持久化
接线，因此它只提供「绑定 + 比对」机制，**不回答信任根从哪来**。
"""

from __future__ import annotations

import pytest

from chain.crypto_adapter import Signer
from chain.identity import (IdentityRegistry, IDENTITY_OK, IDENTITY_MISMATCH,
                            IDENTITY_UNREGISTERED, IDENTITY_UNAVAILABLE)
from chain.models import Credential, EvaluationRecord
from chain.recorder import MatchRecorder
from chain.verify import verify_credential


@pytest.fixture(scope="module")
def registry():
    r = IdentityRegistry()
    if not r.available:
        pytest.skip("真实身份合约不可用（链路径缺失或导入失败）")
    return r


def test_register_then_check_passes(registry):
    s = Signer(signer_id="arena-judge-test-0")
    res = registry.register_signer(s)
    assert res.get("success") is True, res
    status, msg = registry.check("arena-judge-test-0", s.public_key_hex)
    assert status == IDENTITY_OK, msg


def test_impostor_public_key_is_rejected(registry):
    """另一对密钥冒充同一 agent_id → mismatch（挡住冒名顶替换公钥）。"""
    s = Signer(signer_id="arena-judge-test-1")
    assert registry.register_signer(s).get("success") is True
    impostor = Signer(signer_id="impostor-1")
    status, msg = registry.check("arena-judge-test-1", impostor.public_key_hex)
    assert status == IDENTITY_MISMATCH, msg


def test_duplicate_registration_is_refused(registry):
    """重复注册同一 agent_id 必须失败（实测合约返回 success=False）。"""
    s = Signer(signer_id="arena-judge-test-2")
    assert registry.register_signer(s).get("success") is True
    again = registry.register("arena-judge-test-2",
                              Signer(signer_id="other").public_key_hex)
    assert again.get("success") is False, "重复注册竟成功 → 冒名可换公钥"


def test_unregistered_agent_is_not_silently_accepted(registry):
    s = Signer(signer_id="never-registered")
    status, _ = registry.check("never-registered", s.public_key_hex)
    assert status == IDENTITY_UNREGISTERED


def test_invalid_public_key_registration_fails_loudly(registry):
    """非法公钥格式必须返回失败，不得被当成「已锚定」。"""
    res = registry.register("bad-key-agent", "not-a-key")
    assert res.get("success") is False


def test_unavailable_registry_does_not_false_fail(registry_unavailable_check=None):
    """身份合约不可用时：不做校验，但**不得据此判不可信**。

    若把「没做校验」计入 ok，要么把缺失误读为失败，要么（更糟）被读成
    「校验通过」。故该步不参与 ok 判定，只写进 reasons。
    """
    reg = IdentityRegistry(chain_path="__nonexistent__")
    assert not reg.available
    status, _ = reg.check("whoever", "deadbeef")
    assert status == IDENTITY_UNAVAILABLE


def test_forged_credential_caught_once_identity_anchored(registry):
    """P1 留下的缺口被补上：冒名者自造密钥签发的自洽凭证不再被判可信。

    对照 —— 不传身份注册表时，同一份凭证验签是**通过**的（即 P1 的已知边界）。
    """
    judge = Signer(signer_id="arena-judge-test-3")
    assert registry.register_signer(judge).get("success") is True

    record = EvaluationRecord(
        match_id="probe-1", puzzle_id="p1", participants={"q": "x"},
        rounds=1, metrics={"f1": 0.5}, transcript_hash="ab" * 32,
        verdict={"score": 50}, created_at="2026-10-13T00:00:00",
    )
    payload = record.to_payload()
    digest = record.digest()

    # 冒名者用自己的密钥签发，但声称自己是已登记的裁判
    impostor = Signer(signer_id="impostor-3")
    forged = Credential(
        record=payload, record_hash=digest,
        signature_hex=impostor.sign_hex(record.canonical_bytes()),
        public_key_hex=impostor.public_key_hex,
        signer_id="arena-judge-test-3",  # ← 冒名
        algorithm=impostor.algorithm, key_custody=impostor.key_custody,
    )

    # 无身份锚定：验签通过（P1 已知边界，不是 bug）
    plain = verify_credential(forged)
    assert plain.ok is True, "未接身份时冒名凭证本就验签通过（P1 已知边界）"

    # 有身份锚定：必须被拦下
    checked = verify_credential(forged, identity=registry)
    assert checked.ok is False, "冒名凭证未被身份校验拦下"
    assert checked.identity_status == IDENTITY_MISMATCH
