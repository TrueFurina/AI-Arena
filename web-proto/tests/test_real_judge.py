"""A-7 真实 LLM 裁判层的守护测试（**零 API 调用**，全部用假裁判/打桩）。

关键断言对应的设计条款：

  · `test_failure_maps_to_absent_not_zero`
      —— 调用失败必须映射为**缺席**，绝不能折算成 0 分混进聚合
         （design §3；这是本项目反复栽的「静默 fallback 比报错危险」）
  · `test_g1_criterion_is_not_degenerate`
      —— G1 的基线必须是「任取一个裁判」的误差，而不是等权聚合本身；
         否则判据恒为 0/N（sanity 阶段实测抓出，见 design §6-A）
  · `test_node_does_not_repeat_api_call`
      —— 同用例重复访问必须命中缓存（付费 API 实验的生命线）
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judges.llm_panel import (  # noqa: E402
    JudgeUnavailable, LLMJudgeNode, RealJudgePanel, call_judge)
from players.llm import LLMError  # noqa: E402
from eval_real_judges import aggregate_run, rmse  # noqa: E402


class FakePuzzle:
    def __init__(self, soup_id: str, solution: str):
        self.soup_id = soup_id
        self.solution = solution
        self.title = soup_id


class FakeJudge:
    """可控假裁判：scores 给定分数表，或对指定用例抛不可用。"""

    def __init__(self, jid: str, table=None, fail_on=()):
        self.id = jid
        self.model = "fake"
        self.table = table or {}
        self.fail_on = set(fail_on)
        self.calls = 0

    def score(self, puzzle, guess):
        self.calls += 1
        key = (puzzle.soup_id, len(guess))
        if key in self.fail_on:
            raise JudgeUnavailable(f"{self.id} 模拟失败")
        return self.table.get(key, 50.0), "fake"

    def reset_rng(self):
        pass


def _puzzles():
    # 真值 = char_f1(guess, solution)×100；这里直接构造前缀 guess，
    # 让真值随前缀变长而升高（与实验脚本同一构造方式）
    p1 = FakePuzzle("p1", "一个人在餐厅点了一碗海龟汤，喝完就出门开枪自杀，原因是GF替他点错了汤")
    p2 = FakePuzzle("p2", "一个男人在雨中撑伞行走，路遇三个路人分别是老人孕妇和富翁，他一一帮助")
    return p1, p2


# --------------------------------------------------------------------------
# 失败 → 缺席，而非 0 分
# --------------------------------------------------------------------------

def test_failure_maps_to_absent_not_zero():
    p1, _ = _puzzles()
    g = p1.solution[:20]
    a = FakeJudge("a", {(p1.soup_id, len(g)): 10.0})
    b = FakeJudge("b", {(p1.soup_id, len(g)): 90.0}, fail_on=[(p1.soup_id, len(g))])
    panel = RealJudgePanel([a, b], model="fake")
    panel.calibration = {"a": 10.0, "b": 5.0}

    scored = panel.score_all(p1, g)
    assert [v.judge_id for v in scored] == ["a"], "失败的裁判不得产出分数"
    assert panel.last_absent == ["b"]
    assert all(v.score != 0.0 for v in scored), "缺席 ≠ 打 0 分"


def test_all_failed_yields_no_scores():
    p1, _ = _puzzles()
    g = p1.solution[:15]
    a = FakeJudge("a", fail_on=[(p1.soup_id, len(g))])
    b = FakeJudge("b", fail_on=[(p1.soup_id, len(g))])
    panel = RealJudgePanel([a, b], model="fake")
    panel.calibration = {"a": 1.0, "b": 1.0}
    assert panel.score_all(p1, g) == []
    assert sorted(panel.last_absent) == ["a", "b"]


# --------------------------------------------------------------------------
# 标定：覆盖率不足 → unusable（不折算分数）
# --------------------------------------------------------------------------

def test_calibrate_safe_marks_low_coverage_unusable():
    p1, p2 = _puzzles()
    cases = [(p1, p1.solution[:10]), (p1, p1.solution[:20]),
             (p2, p2.solution[:10]), (p2, p2.solution[:20])]
    good = FakeJudge("good")
    flaky = FakeJudge("flaky", fail_on=[(p1.soup_id, 10), (p1.soup_id, 20),
                                        (p2.soup_id, 10)])
    panel = RealJudgePanel([good, flaky], model="fake")
    rmse_out = panel.calibrate_safe(cases, min_coverage=0.5)

    assert "good" in rmse_out
    assert "flaky" not in rmse_out, "覆盖率 0.25 < 0.5 应判不可用"
    assert panel.unusable == ["flaky"]
    # 不可用裁判在评分时也必须缺席
    panel.score_all(p1, p1.solution[:10])
    assert "flaky" in panel.last_absent


def test_calibrate_safe_raises_when_everyone_unavailable():
    p1, _ = _puzzles()
    cases = [(p1, p1.solution[:10])]
    bad = FakeJudge("bad", fail_on=[(p1.soup_id, 10)])
    panel = RealJudgePanel([bad], model="fake")
    with pytest.raises(ValueError):
        panel.calibrate_safe(cases)


# --------------------------------------------------------------------------
# G1 判据非退化（sanity 抓出的退化固化在此）
# --------------------------------------------------------------------------

def test_g1_criterion_is_not_degenerate():
    """G1 主判据是 RMSE 比较（§6-A 修订），此处用异号误差构造验证其能识别集成收益。

    若把基线误用 mean(score)-truth（=等权聚合本身），或把主判据退化成
    「逐用例绝对误差」，本测试必失败（同号偏差时后者恒判无改善）。
    """
    p1, p2 = _puzzles()
    cases = []
    for p in (p1, p2):
        for frac in (0.3, 0.6, 0.9):
            cases.append((p, p.solution[: int(len(p.solution) * frac)]))
    # 两名裁判误差**异号**（一个偏高、一个偏低）：集成可抵消随机分量
    hi = FakeJudge("hi")
    lo = FakeJudge("lo")
    for p, g in cases:
        truth = aggregate_run.__globals__["char_f1"](g, p.solution) * 100
        hi.table[(p.soup_id, len(g))] = min(100.0, truth + 30)
        lo.table[(p.soup_id, len(g))] = max(0.0, truth - 30)
    panel = RealJudgePanel([hi, lo], model="fake")
    panel.calibration = {"hi": 20.0, "lo": 20.0}

    res = aggregate_run(panel, cases)
    assert res["n_scored"] == len(cases)
    assert res["g1_pass_rmse"] is True, "等权 RMSE 应显著低于单点基线"
    assert res["rmse_equal"] < res["rmse_single_mean"]
    for r in res["rows"]:
        assert r["abs_by_judge"], "缓存行必须带各裁判误差，否则续跑会重算"


def test_g1_would_fail_when_judges_share_same_bias():
    """反向对照：同向偏差下集成把偏差减半但**不是**零 —— 判据仍应判为改善
    （RMSE 口径），若改成逐用例绝对误差口径则会误判为无改善。"""
    p1, _ = _puzzles()
    cases = [(p1, p1.solution[: int(len(p1.solution) * f)]) for f in (0.4, 0.8)]
    a = FakeJudge("a")
    b = FakeJudge("b")
    for p, g in cases:
        truth = aggregate_run.__globals__["char_f1"](g, p.solution) * 100
        a.table[(p.soup_id, len(g))] = min(100.0, truth + 30)
        b.table[(p.soup_id, len(g))] = min(100.0, truth + 10)
    panel = RealJudgePanel([a, b], model="fake")
    panel.calibration = {"a": 20.0, "b": 12.0}
    res = aggregate_run(panel, cases)
    assert res["g1_pass_rmse"] is True


def test_rmse_helper():
    import math
    assert rmse([3.0, 4.0]) == pytest.approx(math.sqrt((9 + 16) / 2))
    assert rmse([]) != rmse([])  # NaN


# --------------------------------------------------------------------------
# 缓存：同用例不重复调用端点（付费实验的生命线）
# --------------------------------------------------------------------------

def test_node_does_not_repeat_api_call(monkeypatch):
    calls = {"n": 0}

    def fake_call(model, system, conversation, **kw):
        calls["n"] += 1
        return '{"score": 42, "rationale": "ok"}'

    monkeypatch.setattr("judges.llm_panel._call", fake_call)
    p1, _ = _puzzles()
    node = LLMJudgeNode("strict", "fake-model")
    s1 = node.score(p1, "abc")
    s2 = node.score(p1, "abc")
    assert s1 == s2 == (42.0, "ok")
    assert calls["n"] == 1, "同用例重复评分必须命中缓存（否则重复烧钱）"
    assert node.n_calls == 1


def test_node_caches_failure_too(monkeypatch):
    calls = {"n": 0}

    def flaky_call(model, system, conversation, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return '{"score": 50, "rationale": "r"}'

    monkeypatch.setattr("judges.llm_panel._call", flaky_call)
    p1, _ = _puzzles()
    node = LLMJudgeNode("lenient", "fake-model", retries=1)
    with pytest.raises(JudgeUnavailable):
        node.score(p1, "xyz")
    with pytest.raises(JudgeUnavailable):
        node.score(p1, "xyz")     # 第二次仍失败（缓存了失败，不重试烧钱）
    assert calls["n"] == 1


def test_node_rejects_unparsable_and_bad_score(monkeypatch):
    p1, _ = _puzzles()

    monkeypatch.setattr("judges.llm_panel._call",
                        lambda *a, **k: "这里没有 JSON")
    with pytest.raises(JudgeUnavailable):
        LLMJudgeNode("strict", "fake-model").score(p1, "q")

    monkeypatch.setattr("judges.llm_panel._call",
                        lambda *a, **k: '{"rationale": "无分数"}')
    with pytest.raises(JudgeUnavailable):
        LLMJudgeNode("strict", "fake-model").score(p1, "q2")


def test_node_rejects_unknown_persona():
    with pytest.raises(ValueError):
        LLMJudgeNode("grumpy", "fake-model")


def test_call_judge_falls_back_when_extra_body_rejected(monkeypatch):
    """端点不认 extra_body 时应放大预算重试，而不是直接判不可用。"""
    seen = []

    def fake_call(model, system, conversation, **kw):
        seen.append(kw.get("extra_body"))
        if kw.get("extra_body"):
            # 真实 _call 会把端点异常包成 LLMError 再抛（见 players/llm.py）
            raise LLMError("unsupported parameter")
        return '{"score": 1, "rationale": ""}'

    monkeypatch.setattr("judges.llm_panel._call", fake_call)
    out = call_judge("zai/glm-x", "s", [{"role": "user", "content": "u"}],
                     temperature=0.0, max_tokens=100)
    assert "score" in out
    assert len(seen) == 2 and seen[1] is None