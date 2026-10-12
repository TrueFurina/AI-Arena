"""多裁判共识（P2-A）协议守护 + 实测结论固化。

阈值来源：**跨 10 seeds 实测后留余量**，不是凭单次结果拍的。

  regime    | 单点(均)   | 等权       | 加权       | 单点−加权最小 | 等权−加权最小 | R
  同质       | 7.47~8.85 | 3.22~4.45 | 3.06~4.55 | +3.863      | **−1.336**  | 1.027~1.175
  异质       | 12.68~17.29| 6.95~9.64| 6.39~7.97 | +5.524      | +0.074      | 1.436~1.462

由此确定断言与**刻意不断言**的项：
  ✅ 加权/等权 相对单点的增益：两个 regime 最小 +3.863 → 断言 > 2.0（留足余量）
  ✅ R 的分离度：同质 < 1.25（实测 max 1.175）/ 异质 > 1.35（实测 min 1.436）
  ❌ 「加权优于等权」**不断言**：同质组最小 −1.336，会写出随机失败的假测试
  ⚠️ 改为断言「加权增量只占小头」（实测 14.3%），见 test_known_limit_*
"""

from __future__ import annotations

import math
import statistics as st
import pytest

from engine.puzzles import load_puzzles
from judges.biased import (JudgeProfile, BiasedJudge,
                           make_homogeneous_panel, make_heterogeneous_panel)
from judges.panel import JudgePanel
from eval_judge_consensus import (GUESS_FRACS, make_cases, split_by_puzzle,
                                  rmse_aggregate, rmse_single)

N_JUDGES = 4
N_CALIB_PUZZLES = 6


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------

def _puzzles():
    return load_puzzles()


def _split(puzzles, n_calib=N_CALIB_PUZZLES):
    ids = [p.soup_id for p in puzzles]
    cases = make_cases(puzzles)
    return split_by_puzzle(cases, set(ids[:n_calib]))


def _panel(kind="hetero", seed=0, sign=False):
    mk = (make_homogeneous_panel if kind == "homo"
          else make_heterogeneous_panel)
    return JudgePanel(mk(N_JUDGES, seed0=seed), sign=sign)


def _errors(panel, cases, weights):
    """返回 (单点均值RMSE, 等权RMSE, 加权RMSE)。"""
    single = rmse_single(panel, cases)
    return (st.mean(single.values()),
            rmse_aggregate(panel, weights, cases, "equal"),
            rmse_aggregate(panel, weights, cases, "weighted"))


# --------------------------------------------------------------------------
# A. 协议正确性（违反即结论无效）
# --------------------------------------------------------------------------

def test_calibration_and_eval_are_disjoint_by_puzzle():
    """标定集与评测集必须按题 disjoint，否则「标定」= 用被测数据标自己。"""
    puzzles = _puzzles()
    calib, ev = _split(puzzles)
    assert calib and ev
    c_ids = {c[0].soup_id for c in calib}
    e_ids = {c[0].soup_id for c in ev}
    assert not (c_ids & e_ids), f"标定/评测题目重叠：{c_ids & e_ids}"


def test_guess_never_equals_full_solution():
    """guess 不得取完整 solution——那会让真值=100，分数被 [0,100] 截断后
    噪声只剩半边，RMSE 被系统性低估，且同质裁判会被估出假差异（R 假性>1）。

    实测踩过：取整段 solution 时 4 个 σ=8 的同质裁判 RMSE 竟为
    6.07/4.36/5.07/3.27（全部低于 8），R=1.18 而非 1.0。
    """
    puzzles = _puzzles()
    calib, _ = _split(puzzles)
    for p, guess in calib:
        assert guess != p.solution, f"{p.soup_id} 的 guess 等于完整 solution"
    assert max(GUESS_FRACS) < 1.0


def test_consensus_requires_calibration():
    panel = _panel("hetero", sign=False)
    p = _puzzles()[0]
    with pytest.raises(ValueError, match="标定"):
        panel.consensus_verdict(p, p.solution[:10])


def test_weights_formula_matches_chain_settled_weight():
    """权重必须落在链的 settled-weight 区间 (1.0, 1.5]。"""
    w = JudgePanel.weights_from_rmse({"a": 1.0, "b": 2.0, "c": 10.0})
    assert all(1.0 < v <= 1.5 for v in w.values()), w
    # 误差越小权重越大
    assert w["a"] > w["b"] > w["c"]


def test_homogeneous_judges_yield_near_unit_bandwidth():
    """同质裁判的 R 应接近 1（有限标定样本带来的偏离即「噪声地板」）。"""
    puzzles = _puzzles()
    calib, _ = _split(puzzles)
    panel = _panel("homo", sign=False)
    rmse = rmse_single(panel, calib)
    R = JudgePanel.bandwidth_ratio(JudgePanel.weights_from_rmse(rmse))
    assert R < 1.25, f"同质裁判 R={R:.3f} 偏高 → 权重可能吃到了非异质性信号"


def test_heterogeneous_judges_yield_large_bandwidth():
    puzzles = _puzzles()
    calib, _ = _split(puzzles)
    panel = _panel("hetero", sign=False)
    rmse = rmse_single(panel, calib)
    R = JudgePanel.bandwidth_ratio(JudgePanel.weights_from_rmse(rmse))
    assert R > 1.35, f"异质裁判 R={R:.3f} 偏低 → 标定未分辨出可靠性差异"


def test_fault_tolerance_matches_closed_form():
    """f_max = R/(R+2)，等权时应精确为 1/3（与标准 PBFT 一致）。"""
    assert abs(JudgePanel.fault_tolerance(1.0) - 1 / 3) < 1e-12
    assert abs(JudgePanel.fault_tolerance(1.5) - 1.5 / 3.5) < 1e-12


# --------------------------------------------------------------------------
# B. 共识拒绝行为 —— 「移除链就没有这个数字」的核心
# --------------------------------------------------------------------------

def test_consensus_refuses_to_publish_when_reliable_weight_missing():
    """缺席高可靠裁判 → 响应权重不足 → 拒绝发布（score 必为 None）。"""
    puzzles = _puzzles()
    panel = _panel("hetero", sign=False)
    calib, _ = _split(puzzles)
    panel.calibration = rmse_single(panel, calib)
    w = JudgePanel.weights_from_rmse(panel.calibration)
    top2 = sorted(w, key=lambda k: -w[k])[:2]

    p = puzzles[-1]
    guess = p.solution[:max(1, int(len(p.solution) * 0.5))]
    v = panel.consensus_verdict(p, guess, absent=top2)
    assert not v.accepted, "权重不足却发布了结论"
    assert v.score is None, "拒绝发布时必须不给分数（不得给凑合的数字）"
    assert v.detail["responded_ratio"] <= 2 / 3


def test_consensus_publishes_when_only_weak_judge_absent():
    """缺席最低可靠裁判 → 剩余权重仍充足 → 正常发布。"""
    puzzles = _puzzles()
    panel = _panel("hetero", sign=False)
    calib, _ = _split(puzzles)
    panel.calibration = rmse_single(panel, calib)
    w = JudgePanel.weights_from_rmse(panel.calibration)
    worst = min(w, key=lambda k: w[k])

    p = puzzles[-1]
    guess = p.solution[:max(1, int(len(p.solution) * 0.5))]
    v = panel.consensus_verdict(p, guess, absent=[worst])
    assert v.accepted, f"仅缺席最弱裁判却拒绝发布（权重比 {v.detail}）"
    assert v.score is not None


def test_no_judge_responding_is_rejected():
    puzzles = _puzzles()
    panel = _panel("hetero", sign=False)
    calib, _ = _split(puzzles)
    panel.calibration = rmse_single(panel, calib)
    v = panel.consensus_verdict(puzzles[0], "x", absent=panel.judge_ids)
    assert not v.accepted and v.score is None
    assert v.detail["reason"]


def test_consensus_refusal_holds_when_chain_unavailable():
    """链不可用 → 降级路径必须保持同一拒绝行为（否则环境一换结论就变）。

    本机有真实链，上面的拒绝测试走的是 CW-PBFT 引擎路径；本条强制
    chain_path 不存在，走本地 2/3 权重门限。**两条路径的拒绝行为必须
    一致**——变异验证：把降级门限 ratio > 2/3 改成 ratio > 0，本条应变红
    （曾实测：不加本条时该变异全绿，因为真实链路径遮蔽了降级路径）。
    """
    puzzles = _puzzles()
    panel = JudgePanel(make_heterogeneous_panel(N_JUDGES, seed0=0),
                       sign=False, chain_path="__nonexistent__")
    assert not panel.consensus_available, "链路径不存在却报告共识引擎可用"
    calib, _ = _split(puzzles)
    panel.calibration = rmse_single(panel, calib)
    w = JudgePanel.weights_from_rmse(panel.calibration)
    top2 = sorted(w, key=lambda k: -w[k])[:2]

    p = puzzles[-1]
    guess = p.solution[:max(1, int(len(p.solution) * 0.5))]
    v = panel.consensus_verdict(p, guess, absent=top2)
    assert v.backend == "local-weighted", f"未走降级路径：{v.backend}"
    assert not v.accepted and v.score is None
    # 全在场时应正常发布（降级路径不是一律拒绝）
    v2 = panel.consensus_verdict(p, guess)
    assert v2.accepted and v2.score is not None


# --------------------------------------------------------------------------
# C. 签名（每个裁判独立签名，可离线验签）
# --------------------------------------------------------------------------

def test_each_verdict_is_independently_signed_and_verifiable():
    puzzles = _puzzles()
    panel = JudgePanel(make_heterogeneous_panel(3, seed0=0), sign=True)
    vs = panel.score_all(puzzles[0], puzzles[0].solution[:12])
    assert len(vs) == 3
    for v in vs:
        assert v.signature_hex and v.public_key_hex
        assert v.verify() is True


def test_tampered_verdict_score_fails_verification():
    puzzles = _puzzles()
    panel = JudgePanel(make_heterogeneous_panel(2, seed0=0), sign=True)
    v = panel.score_all(puzzles[0], puzzles[0].solution[:12])[0]
    v.score = 99.9  # 篡改分数（签名不变）
    assert v.verify() is False


# --------------------------------------------------------------------------
# D. 实测结论（跨 seed 稳健项才断言）
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["homo", "hetero"])
def test_consensus_beats_single_judge_across_seeds(kind):
    """加权共识相对单点裁判的增益：跨 3 seeds 全为正（实测最小 +3.863）。"""
    puzzles = _puzzles()
    calib, ev = _split(puzzles)
    gains = []
    for seed in (0, 1, 2):
        panel = _panel(kind, seed=seed * 100, sign=False)
        weights = JudgePanel.weights_from_rmse(rmse_single(panel, calib))
        s_mean, _eq, w_rmse = _errors(panel, ev, weights)
        gains.append(s_mean - w_rmse)
    assert all(g > 2.0 for g in gains), (
        f"{kind}: 加权相对单点增益 {[round(g, 3) for g in gains]}，"
        "存在 <= 2.0 的种子")


@pytest.mark.parametrize("kind", ["homo", "hetero"])
def test_ensemble_beats_single_judge_even_without_weighting(kind):
    """等权集成本身就大幅优于单点（实测：异质 17.3→8.2）。"""
    puzzles = _puzzles()
    calib, ev = _split(puzzles)
    gains = []
    for seed in (0, 1, 2):
        panel = _panel(kind, seed=seed * 100, sign=False)
        weights = JudgePanel.weights_from_rmse(rmse_single(panel, calib))
        s_mean, eq, _w = _errors(panel, ev, weights)
        gains.append(s_mean - eq)
    assert all(g > 1.5 for g in gains), f"{kind}: 等权集成增益 {gains}"


# --------------------------------------------------------------------------
# E. 已知局限（真正改善后应失败并更新，而非删断言）
# --------------------------------------------------------------------------

def test_known_limit_weighting_contributes_minority_of_gain():
    """权重带来的增量只占总增益的小头（实测 14.3%）。

    主体收益来自「多裁判集成」而非「贡献加权」——这与真实链自身的结论
    方向一致（R≈1.007 时 CW-PBFT ≡ 标准 PBFT）。若将来放开权重带宽、
    加权增量占比显著上升，本断言会失败并提醒更新结论。

    ⚠️ 附带的诚实记录（不写成断言，因为它是「不稳健」而非「恒假」）：
    同质裁判下「加权优于等权」跨 10 seeds 的最小值为 **−1.336**（负值），
    故本套件刻意不断言该项——断言它会得到一个随机失败的假测试。
    """
    puzzles = _puzzles()
    calib, ev = _split(puzzles)
    shares = []
    for seed in (0, 1, 2):
        panel = _panel("hetero", seed=seed * 100, sign=False)
        weights = JudgePanel.weights_from_rmse(rmse_single(panel, calib))
        s_mean, eq, w_rmse = _errors(panel, ev, weights)
        ensemble_gain = s_mean - eq
        weight_gain = eq - w_rmse
        total = ensemble_gain + weight_gain
        if total > 0:
            shares.append(weight_gain / total)
    assert shares, "总增益非正，无法计算占比"
    assert max(shares) < 0.35, (
        f"加权增量占比 {[round(s, 3) for s in shares]} 已超过 35% → "
        "权重的贡献已不再是小头，请更新本局限描述")
