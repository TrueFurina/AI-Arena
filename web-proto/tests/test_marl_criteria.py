"""P2-B 判据计算守护（train_marl.per_lie_damage + 报告语义）。

DESIGN §6-A 的两条约定在此固化：

1. **恒真主持 → None，不是 0.0**（发现 2）：λ 高于伤害收益时主持理性地
   不说谎，处理未激活。返回 0.0 会被读成「有谎言但零伤害」——混淆两者
   会让 E1' 在惰性区间假装被判定过（多数裁决的分母被污染）。
2. 每谎伤害公式本身（F1 版主判据 + dhit 版辅证共用）。
"""
import pytest

from train_marl import per_lie_damage


def test_damage_formula():
    """每谎伤害 =（truthful 基线 − 该主持指标）/ 谎言数。"""
    assert per_lie_damage(1.0, 0.75, 2) == pytest.approx(0.125)
    assert per_lie_damage(0.5, 0.4, 5) == pytest.approx(0.02)
    assert per_lie_damage(0.15, 0.124, 2) == pytest.approx(0.013)


def test_damage_is_none_when_host_never_lies():
    """恒真主持 → None（处理未激活）；不伪造 0.0。"""
    assert per_lie_damage(1.0, 1.0, 0) is None
    # 主持指标更差但零谎言：仍是「未激活」，不是「零伤害」
    assert per_lie_damage(1.0, 0.9, 0) is None
    # 负数谎言数（不可能输入）同样按未激活处理，不除出怪数
    assert per_lie_damage(1.0, 0.9, -1) is None


def test_damage_monotonic_in_host_damage():
    """同一谎言数下，主持越坏（指标越低）每谎伤害越大——判据方向正确。"""
    worse = per_lie_damage(1.0, 0.5, 2)
    milder = per_lie_damage(1.0, 0.8, 2)
    assert worse > milder > 0


def test_damage_shrinks_with_more_lies_at_same_total_drop():
    """总落差固定时谎言越多，每谎伤害越小（归一化语义）。"""
    assert per_lie_damage(1.0, 0.5, 2) > per_lie_damage(1.0, 0.5, 4)
