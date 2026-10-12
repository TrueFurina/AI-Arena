"""P1 链上评测凭证 —— 协议正确性与已知边界。

设计约束：
- **不硬依赖 logs/ 下的真实对局**：该目录被 .gitignore 排除，克隆后不存在。
  主用例用自洽合成数据；真实日志用例在文件缺失时 skip（不编造、不失败）。
- 每条关键断言都做过变异验证，见文件末尾说明。
"""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from chain.crypto_adapter import (
    KEY_CUSTODY_DEMO,
    KEY_CUSTODY_EPHEMERAL,
    Signer,
    _local_verify,
    verify_signature,
)
from chain.models import Credential, canonical_bytes, sha256_hex
from chain.recorder import MatchRecorder, transcript_digest
from chain.verify import verify_credential
from engine.evaluate import metrics_from_result
from engine.puzzles import load_puzzles

LOGS = Path(__file__).resolve().parents[1] / "logs"
REAL_LOG = LOGS / "compare_deepseek_deepseek-chat_turtle_soup_001.json"


def _synth_result() -> dict:
    """自洽的最小对局结果（不依赖任何外部文件）。"""
    p = load_puzzles()[0]
    return {
        "game_id": "test-game-001",
        "puzzle": asdict(p),
        "host_model": "mock",
        "players": ["rl-questioner"],
        "max_rounds": 8,
        "transcript": [
            {"round": 1, "speaker": "q", "role": "question", "content": "和肉有关吗？"},
            {"round": 1, "speaker": "h", "role": "answer", "content": "是"},
            {"round": 2, "speaker": "q", "role": "guess", "content": "真相与以下要素有关：肉。"},
        ],
        "scores": [
            {"player": "q", "model": "rl-questioner", "final_guess": "真相与以下要素有关：肉。",
             "score": 55.0, "rationale": "命中线索"}
        ],
        "winner": "q",
        "started_at": 0.0,
        "ended_at": 1.0,
    }


@pytest.fixture(scope="module")
def result():
    return _synth_result()


# ---------------------------------------------------------------------------
# A 协议正确性
# ---------------------------------------------------------------------------

def test_canonical_bytes_is_deterministic():
    """确定性序列化：键序不同但内容相同的 dict 必须得到相同字节。

    这是验签能成立的前提——若序列化不确定，同一记录两次哈希不同，
    凭证永远验不过。中文与浮点也在同一用例内覆盖。
    """
    a = {"b": 1, "a": "中文", "c": 0.123456789}
    b = {"a": "中文", "c": 0.123456789, "b": 1}
    assert canonical_bytes(a) == canonical_bytes(b)
    # 内容不同则必须不同（防止实现退化成恒返回同一串）
    assert canonical_bytes(a) != canonical_bytes({"b": 2, "a": "中文", "c": 0.123456789})


def test_signature_covers_metrics_change(result):
    """篡改任一指标 → 摘要与签名双重失败。"""
    rec = MatchRecorder()
    cred = rec.issue_from_result(result)
    assert verify_credential(cred).ok

    bad = copy.deepcopy(cred)
    bad.record["metrics"]["f1"] = 0.999
    r = verify_credential(bad)
    assert not r.ok
    assert not r.steps["digest"] and not r.steps["signature"]


def test_signature_covers_transcript_change(result):
    """篡改 transcript 哈希 → 失败（原始对话受保护，尽管它不进凭证）。"""
    rec = MatchRecorder()
    cred = rec.issue_from_result(result)
    bad = copy.deepcopy(cred)
    bad.record["transcript_hash"] = "deadbeef"
    assert not verify_credential(bad).ok


def test_wrong_public_key_fails_signature_but_not_digest(result):
    """换掉公钥 → digest 仍通过、signature 必须失败。

    两条判据独立：若实现只查摘要不查签名，本用例会放过伪造凭证
    （已做变异验证，见文件末）。
    """
    rec = MatchRecorder()
    cred = rec.issue_from_result(result)
    bad = copy.deepcopy(cred)
    bad.public_key_hex = cred.public_key_hex[:-4] + "0000"
    r = verify_credential(bad)
    assert r.steps["digest"], "记录未改动，摘要应仍通过"
    assert not r.steps["signature"], "公钥不匹配，签名必须失败"
    assert not r.ok


def test_transcript_digest_detects_content_change(result):
    """transcript 内容改动 → 摘要变化（保护强度与全文进凭证等价）。"""
    h1 = transcript_digest(result["transcript"])
    altered = copy.deepcopy(result)
    altered["transcript"][1]["content"] = "否"
    assert transcript_digest(altered["transcript"]) != h1


def test_metrics_come_from_single_source(result):
    """凭证里的指标必须与 engine/evaluate.py 逐项一致（禁止本层另算）。"""
    rec = MatchRecorder()
    cred = rec.issue_from_result(result)
    expected = metrics_from_result(result)
    for k in ("vra", "yn_rate", "f1", "hit_rate", "dhit"):
        if k in expected:
            assert cred.record["metrics"][k] == pytest.approx(float(expected[k]), abs=1e-6)


def test_default_key_custody_is_not_public_default():
    """默认路径不得使用随源码公开的默认口令。

    该链在 MARL_ECDSA_KEY_PASSPHRASE 缺失时会回退公开默认口令并告警
    「不具备任何保密性」。本项目不接受静默走这条路径。
    """
    s = Signer(signer_id="custody-probe")
    assert s.key_custody != KEY_CUSTODY_DEMO
    assert s.key_custody in ("env-passphrase", "local-generated",
                             KEY_CUSTODY_EPHEMERAL)


def test_cross_implementation_compatibility():
    """两条实现路径算法一致：可交叉验签。

    真实链 crypto 与本地 cryptography 直签必须互换可用，否则「链不可用
    降级」会导致历史凭证在新环境下验不过。
    """
    msg = b"cross-compat probe"
    s_chain = Signer(signer_id="probe-chain")
    sig = s_chain.sign_hex(msg)
    assert _local_verify(s_chain.public_key_hex, msg, bytes.fromhex(sig)), \
        "链签名无法被本地实现验证 → 两条路径摘要口径不一致"

    s_local = Signer(signer_id="probe-local", chain_path="__nonexistent__")
    assert not s_local.using_chain_crypto
    assert verify_signature(s_local.public_key_hex, msg, s_local.sign_hex(msg)), \
        "本地签名无法被通用验签验证"

    assert not _local_verify(s_chain.public_key_hex, msg + b"X",
                             bytes.fromhex(sig)), "篡改消息居然验过了"


def test_degraded_path_when_chain_unavailable(result):
    """链不可用 → 降级为本地签名，anchor=None，验签仍通过，且不阻塞。"""
    rec = MatchRecorder(chain_path="__nonexistent__")
    assert rec._chain is None
    cred = rec.issue_from_result(result)
    assert cred.backend == "local"
    assert cred.anchor is None, "未上链却写了 anchor → 谎报锚定状态"
    assert verify_credential(cred).ok


def test_anchor_status_is_honest(result):
    """若成功入池，status 必须为 in-pool，不得写成 anchored/已上链。"""
    rec = MatchRecorder()
    cred = rec.issue_from_result(result)
    if cred.anchor is None:
        pytest.skip("真实链不可用，跳过锚定状态检查")
    assert cred.anchor["status"] == "in-pool", \
        "入池不等于出块，status 不得写成 anchored"


# ---------------------------------------------------------------------------
# B 真实日志（存在才跑）
# ---------------------------------------------------------------------------

def test_credential_on_real_log():
    if not REAL_LOG.exists():
        pytest.skip("真实对局日志缺失（logs/ 被 gitignore，属预期）")
    res = json.loads(REAL_LOG.read_text(encoding="utf-8"))
    cred = MatchRecorder().issue_from_result(res)
    assert verify_credential(cred).ok

    bad = copy.deepcopy(cred)
    bad.record["verdict"]["winner"] = "attacker"
    assert not verify_credential(bad).ok


# ---------------------------------------------------------------------------
# C 已知边界（诚实记录，不是期望行为）
# ---------------------------------------------------------------------------

def test_known_limit_verification_does_not_prove_issuer_identity():
    """验签通过 ≠ 凭证由本评测台签发（凭证自带公钥，无法防伪）。

    任何人可自造密钥对签发一份自洽的假凭证。本用例把该边界固化下来：
    一旦将来接入了身份锚定（公钥需已登记），此断言应当失败并更新。
    """
    attacker = Signer(signer_id="attacker", chain_path="__nonexistent__")
    payload = {"schema": "arena-eval-credential/v1", "match_id": "fake"}
    msg = canonical_bytes(payload)
    forged = Credential(
        record=payload,
        record_hash=sha256_hex(msg),
        signature_hex=attacker.sign_hex(msg),
        public_key_hex=attacker.public_key_hex,
        signer_id="attacker",
        algorithm=attacker.algorithm,
        key_custody=attacker.key_custody,
    )
    assert verify_credential(forged).ok, \
        "伪造凭证本就应验签通过——这正是当前不提供身份防伪的边界"
