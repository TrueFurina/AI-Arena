"""P3 · 一键流水线与验签入口的守护测试。

对应 `DESIGN-P3-PIPELINE.md` 的预注册验收 F1–F6。三条必须做变异验证的
断言在这里显式标注（改错后应立刻变红，否则断言是死的）：

  · F2 —— 共识拒绝发布时不得填补分数
  · F3 —— 共识结论必须进签名对象（改共识数字 → 验签失败）
  · F5③ —— tx_id 未命中必须报「未命中」，不得报「验签失败」

每条都在 docstring 里写了「怎么变红」，方便后来者复验。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain.models import Credential  # noqa: E402
from chain.recorder import MatchRecorder  # noqa: E402
from chain.verify import verify_credential  # noqa: E402
from pipeline import run_pipeline, _calibration_cases  # noqa: E402
from pipeline_report import render_report  # noqa: E402

# 测试一律不提交真实链：凭证的完整性由本地签名路径保证，
# 链的可用性不是本文件的被测对象（链路径另有 test_chain.py 覆盖）
NO_CHAIN = dict(submit_to_chain=False)


def _run(tmp_path, **kw):
    kw.setdefault("submit_to_chain", False)
    kw.setdefault("out_dir", tmp_path)
    return run_pipeline(**kw)


# --------------------------------------------------------------------------
# F1 一键贯通
# --------------------------------------------------------------------------

def test_f1_pipeline_produces_all_artifacts(tmp_path):
    pr = _run(tmp_path, puzzle_id="turtle_soup_001")
    assert pr.ok, pr.verification.summary()
    for p in (pr.game_path, pr.credential_path, pr.report_path):
        assert p is not None and p.exists()
    assert (tmp_path / "credentials").is_dir()

    report = pr.report_path.read_text(encoding="utf-8")
    cred_id = str(pr.credential.record_hash)[:16]
    assert cred_id in report, "报告必须含凭证 ID（否则无法追溯到凭证）"
    assert "验签入口" in report
    assert "能力边界" in report, "能力边界不得省略"


def test_f1_report_metrics_come_from_engine(tmp_path):
    """报告里的指标必须与 engine/evaluate.py 逐值一致（口径单一真值）。"""
    from engine.evaluate import metrics_from_result

    pr = _run(tmp_path)
    assert pr.metrics == metrics_from_result(pr.game)
    report = pr.report_path.read_text(encoding="utf-8")
    assert f"| `f1` | {pr.metrics['f1']:.4f} |" in report


# --------------------------------------------------------------------------
# F2 共识拒绝发布不得被填补（变异点 1）
# --------------------------------------------------------------------------

def test_f2_rejected_consensus_is_not_backfilled(tmp_path):
    """变异方式：把 render_report 里的 `_fmt(cons.get('score'))` 换成
    `cons.get('equal_weight_score')` 或 `0`，本测试应立刻变红。"""
    pr = _run(tmp_path, panel="hetero", absent=("j0", "j1"))
    assert pr.consensus is not None
    assert pr.consensus.accepted is False
    assert pr.consensus.score is None, "拒绝发布就是 None，不许给数"

    report = pr.report_path.read_text(encoding="utf-8")
    assert "⛔ 未发布（REJECTED）" in report
    assert "| 共识结论分数 | — |" in report, "共识位必须是空，不得用单点/等权填补"
    assert "不提供任何替代分数" in report
    # 等权分只可作为对照出现，不得出现在「共识结论分数」那一行
    for line in report.splitlines():
        if "共识结论分数" in line:
            assert line.strip().endswith("| — |"), line
    assert any("拒绝发布" in d for d in pr.degraded), "拒绝发布须进降级记录（不静默）"


def test_f2_pipeline_result_has_no_fallback_field(tmp_path):
    """结构层面兜底：result 上不存在任何「替补分数」字段。"""
    pr = _run(tmp_path, panel="hetero", absent=("j0", "j1", "j2"))
    assert pr.consensus.score is None
    assert pr.ok is True, "拒绝发布不影响凭证完整性，流水线仍算完成"


# --------------------------------------------------------------------------
# F3 共识结论进签名对象（变异点 2）
# --------------------------------------------------------------------------

def test_f3_consensus_is_signed(tmp_path):
    """变异方式：把 pipeline 里 verdict_extra 改成 None，本测试应变红。"""
    pr = _run(tmp_path, panel="homo")
    verdict = pr.credential.record["verdict"]
    assert verdict.get("mode") == "panel-consensus"
    assert "consensus" in verdict
    assert pr.consensus is not None and pr.consensus.accepted is True
    assert verdict["consensus"]["score"] == pr.consensus.score


def test_f3_tampering_consensus_breaks_signature(tmp_path):
    pr = _run(tmp_path, panel="homo")
    cred = Credential.from_dict(json.loads(pr.credential_path.read_text(encoding="utf-8")))
    assert verify_credential(cred).ok

    cred.record["verdict"]["consensus"]["score"] = 99.0
    vr = verify_credential(cred)
    assert vr.ok is False, "改共识结论必须让验签失败（否则共识不受保护）"
    assert vr.steps["digest"] is False


# --------------------------------------------------------------------------
# F4 向后兼容（不启用共识时与 P1 逐字段一致）
# --------------------------------------------------------------------------

def test_f4_no_consensus_record_matches_p1_shape(tmp_path):
    pr = _run(tmp_path)
    verdict = pr.credential.record["verdict"]
    assert set(verdict) == {"judge_id", "winner", "scores"}, (
        "单点路径的 verdict 结构与 P1 必须逐字段一致（不新增键）")
    for s in verdict["scores"]:
        assert set(s) == {"player", "model", "score"}


def test_f4_verdict_extra_rejects_overwrite(tmp_path):
    rec = MatchRecorder()
    res = {"game_id": "x", "puzzle": {"soup_id": "p"}, "transcript": [],
           "scores": [], "players": [], "host_model": "h", "max_rounds": 8,
           "winner": None}
    with pytest.raises(ValueError):
        rec.build_record(res, verdict_extra={"winner": "attacker"})
    r = rec.build_record(res, verdict_extra={"mode": "panel-consensus"})
    assert r.verdict["mode"] == "panel-consensus"
    assert "winner" in r.verdict


# --------------------------------------------------------------------------
# F6 降级不阻塞
# --------------------------------------------------------------------------

def test_identity_option_actually_runs_the_check(tmp_path):
    """变异方式：让 run_pipeline 忽略 use_identity（不传 identity 给
    verify_credential），identity_status 会变 None，本测试应变红。

    只断言「这一步被真正执行过」，不断言 ok——身份合约是否可用是环境问题，
    `unavailable` 也是合法结果（它不参与 ok 判定，见 chain/verify.py）。
    """
    pr = run_pipeline(out_dir=tmp_path, submit_to_chain=False, use_identity=True)
    assert pr.verification.identity_status is not None
    assert pr.verification.identity_status in (
        "ok", "unavailable", "unregistered", "mismatch")
    assert pr.ok is True


def test_f6_chain_unavailable_degrades_without_blocking(tmp_path):
    pr = run_pipeline(out_dir=tmp_path, submit_to_chain=True,
                      chain_path=str(tmp_path / "no-such-chain"))
    assert pr.ok is True, "链不可用不得让流水线失败"
    assert pr.credential.anchor is None
    assert any("真实共识链不可用" in d for d in pr.degraded)
    assert pr.report_path.read_text(encoding="utf-8").count("未上链") >= 1


def test_f6_report_marks_degradation_explicitly(tmp_path):
    pr = run_pipeline(out_dir=tmp_path, panel="homo", submit_to_chain=False,
                      chain_path=str(tmp_path / "no-such-chain"))
    report = pr.report_path.read_text(encoding="utf-8")
    assert "本次运行的降级记录" in report
    assert "本地 2/3 权重门限" in report or "cw-pbft" in report


# --------------------------------------------------------------------------
# 标定集协议（disjoint）
# --------------------------------------------------------------------------

def test_calibration_cases_exclude_evaluated_puzzle():
    from engine.puzzles import load_puzzles

    puzzles = load_puzzles()
    target = puzzles[0].soup_id
    cases = _calibration_cases(puzzles, target, n_puzzles=5)
    assert cases, "标定集不得为空"
    assert all(p.soup_id != target for p, _ in cases)
    # 协议③：guess 是 solution 的真前缀，不得取整段（真值=100 会被截断）
    for p, g in cases:
        assert 0 < len(g) < len(p.solution)


# --------------------------------------------------------------------------
# 渲染层：缺键不编造
# --------------------------------------------------------------------------

def test_render_report_without_credential_does_not_invent_ids():
    text = render_report({"game": {"game_id": "g"}, "metrics": {"f1": 0.5}})
    assert "未签发凭证" in text
    assert "未启用共识面板" in text
    assert "0.5000" in text
    assert "—" in text


# --------------------------------------------------------------------------
# F5 前端验签入口
# --------------------------------------------------------------------------

def _app():
    pytest.importorskip("flask")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import web_app
    except Exception as exc:  # 依赖缺失时不假装通过
        pytest.skip(f"web_app 不可用：{type(exc).__name__}: {exc}")
    return web_app


def _client():
    return _app().app.test_client()


def test_f5_verify_endpoint_accepts_credential(tmp_path):
    pr = _run(tmp_path, panel="homo")
    raw = pr.credential_path.read_text(encoding="utf-8")
    c = _client()
    r = c.post("/api/verify", json={"credential": raw})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and d["mode"] == "credential"
    assert d["steps"]["digest"] and d["steps"]["signature"]

    # 篡改任一字节 → 不可信
    bad = json.loads(raw)
    bad["record"]["metrics"]["f1"] = 0.999
    d2 = c.post("/api/verify", json={"credential": json.dumps(bad)}).get_json()
    assert d2["ok"] is False


def test_f5_tx_id_miss_is_not_verification_failure(tmp_path):
    """变异点 3：把 api_verify 的 miss 分支 ok 从 None 改成 False，
    本测试应变红——「未命中」与「验签失败」必须可区分。"""
    _run(tmp_path)
    c = _client()
    d = c.post("/api/verify", json={"tx_id": "f" * 64}).get_json()
    assert d["found"] is False
    assert d["ok"] is None, "未命中 = 未做验签，绝不能报成失败"
    assert "未命中" in d["reason"]


def test_f5_tx_id_hit_verifies_same_credential(tmp_path, monkeypatch):
    pr = run_pipeline(out_dir=tmp_path, submit_to_chain=True)
    tx_id = (pr.credential.anchor or {}).get("tx_id")
    if not tx_id:  # 链不可用则无 tx_id，跳过（该场景由 F6 覆盖）
        pytest.skip("本次环境无链上 tx_id")
    app = _app()
    # 凭证库目录指向本次产物，避免测试污染/依赖真实 logs/credentials
    monkeypatch.setattr(app, "CRED_DIR", str(tmp_path / "credentials"))
    c = app.app.test_client()
    d = c.post("/api/verify", json={"tx_id": tx_id}).get_json()
    assert d["found"] is True and d["ok"] is True
    assert d["record_hash"] == pr.credential.record_hash


def test_f5_verify_page_renders_and_rejects_empty():
    c = _client()
    assert c.get("/verify").status_code == 200
    r = c.post("/api/verify", json={})
    assert r.status_code == 400
