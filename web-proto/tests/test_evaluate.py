"""指标口径锁定测试（engine/evaluate.py）。

这些用例的目的不是覆盖率，而是**防口径漂移**：
历史结论（B 方案 DD=0.53 / GG=0.29、cross-play 析因）全部依赖这里的函数，
一旦有人改动 char_f1 / metrics_from_result 的实现而不同步结论，这些用例
必须 FAIL。变异验证：故意改坏实现应立刻变红。
"""
import glob
import json
import os

import pytest

from engine.evaluate import (
    char_f1, is_valid_question, yn_rate, metrics_from_result, Evaluator,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG_DIR = os.path.join(ROOT, "logs")


# ---------- 基础性质 ----------
def test_char_f1_symmetric_and_bounded():
    assert char_f1("abc", "abc") == pytest.approx(1.0)
    assert char_f1("abc", "xyz") == pytest.approx(0.0)
    assert char_f1("ab", "bcd") == pytest.approx(char_f1("bcd", "ab"))
    assert 0.0 <= char_f1("abc", "abd") <= 1.0
    assert char_f1("", "abc") == 0.0


def test_is_valid_question():
    assert is_valid_question("他是自杀的吗？")
    assert is_valid_question("为什么他会死？")
    assert not is_valid_question("他是自杀的")
    assert not is_valid_question("")


def test_yn_rate():
    assert yn_rate(["是", "否", "无关"]) == pytest.approx(2 / 3)
    assert yn_rate([]) == 0.0


def test_metrics_from_result_shapes():
    res = {
        "transcript": [
            {"role": "question", "content": "他是自杀的吗？"},
            {"role": "answer", "content": "是"},
            {"role": "question", "content": "陈述句"},
            {"role": "answer", "content": "无关"},
            {"role": "guess", "content": "abc"},
        ],
        "puzzle": {"solution": "abd"},
    }
    m = metrics_from_result(res)
    assert m["total_q"] == 2 and m["valid_q"] == 1
    assert m["vra"] == pytest.approx(0.5)
    assert m["yn_rate"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(char_f1("abc", "abd"))


# ---------- 历史口径锁 ----------
def _mean_f1(pattern):
    paths = sorted(glob.glob(os.path.join(LOG_DIR, pattern)))
    if not paths:
        return None
    vals = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            vals.append(metrics_from_result(json.load(f))["f1"])
    return sum(vals) / len(vals)


@pytest.mark.parametrize("pattern,expected,tag", [
    ("compare_deepseek_deepseek-chat_*.json", 0.53, "DD"),
    ("compare_zai_glm-4-flash_*.json", 0.29, "GG"),
])
def test_history_f1_locked(pattern, expected, tag):
    """用 B 方案真实日志锁定 F1 口径（对局日志在 .gitignore 内，缺失则跳过）。

    若 CI/新克隆环境无日志，跳过而非编造；本地有日志时必须与历史结论一致。
    """
    got = _mean_f1(pattern)
    if got is None:
        pytest.skip(f"历史对局日志缺失（logs/ 被 .gitignore 排除）：{pattern}")
    assert got == pytest.approx(expected, abs=0.02), (
        f"{tag} 平均 F1={got:.3f} 与历史结论 {expected} 不符 → 指标口径可能已漂移"
    )


# ---------- 矩阵与析因 ----------
def _rec(f1):
    """手工构造的指标记录：必须补齐 METRIC_KEYS 全字段。

    Evaluator.agg() 走严格契约（缺字段直接 KeyError，不做静默补 0）——
    把缺失指标当 0 平均会让均值静默失真，比报错危险得多。
    """
    return {"vra": 1.0, "yn_rate": 1.0, "f1": f1, "hit_rate": 1.0,
            "valid_q": 1, "total_q": 1}


def test_evaluator_matrix_and_factorial():
    ev = Evaluator()
    ev.add("A", "X", [_rec(0.50)])
    ev.add("A", "Y", [_rec(0.40)])
    ev.add("B", "X", [_rec(0.30)])
    ev.add("B", "Y", [_rec(0.20)])

    assert ev.agg("A", "X")["f1"] == pytest.approx(0.50)
    assert ev.matrix("f1", ["A", "B"], ["X", "Y"]) == [
        pytest.approx([0.50, 0.40]), pytest.approx([0.30, 0.20]),
    ]

    fac = ev.factorial_2x2("A", "B", "X", "Y", metric="f1")
    assert fac["row_effect_at_c0"] == pytest.approx(0.20)
    assert fac["row_effect_at_c1"] == pytest.approx(0.20)
    assert fac["col_effect_at_r0"] == pytest.approx(0.10)
    assert fac["interaction"] == pytest.approx(0.0)
    assert fac["diag_diff"] == pytest.approx(0.30)
    assert fac["cross_diff"] == pytest.approx(0.10)
    assert fac["additive_gap"] == pytest.approx(0.20)
    assert fac["dominant"] == "row"


def test_evaluator_agg_requires_data():
    ev = Evaluator()
    with pytest.raises(KeyError):
        ev.agg("A", "X")
