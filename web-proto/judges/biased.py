"""受控偏差裁判（零成本）——用于在**已知 ground truth** 的环境下比较聚合方法。

## 这是什么 / 不是什么

**是**：一个测量误差模型。裁判分数 = 客观质量 × 尺度 + 系统截距 + 噪声。
      用途是在 ground truth 已知时，量化「加权共识 vs 单点裁判 vs 等权平均」
      谁更贴近客观值，且**完全零成本、可复现**。

**不是**：对真实 LLM 裁判行为的建模。真实裁判的偏差形态未知，本模块
      不宣称任何关于真实裁判的结论。真实标定需付费 API，列为后续工作。

## 三个参数分别对应一类真实偏差

  slope     尺度偏差——对质量差异的敏感度不同（有人打分压缩、有人放大）
  intercept 系统截距——**自偏好就长这样**：一律给自己人加固定分（续5 实测）
  sigma     随机噪声——同一份答案重复评也会抖

## 为什么要固定种子

噪声必须是**可复现**的，否则「加权优于等权」的结论可能只是某次抽样的运气。
所有随机性走 `random.Random(seed)`，给定 seed 结果逐位一致。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Optional

from players.base import Judge

# 分数取值区间（与 Judge.score 契约一致：0-100）
SCORE_MIN, SCORE_MAX = 0.0, 100.0


@dataclass(frozen=True)
class JudgeProfile:
    """一个裁判的偏差参数。frozen 保证标定期间参数不被意外改动。"""

    judge_id: str
    slope: float = 1.0
    intercept: float = 0.0
    sigma: float = 0.0
    seed: int = 0

    def describe(self) -> str:
        return (f"{self.judge_id}: slope={self.slope:.2f} "
                f"intercept={self.intercept:+.1f} sigma={self.sigma:.1f}")


class BiasedJudge(Judge):
    """按测量误差模型打分的裁判。

    客观基准取 `char_f1(guess, solution) * 100`——这是本项目已确立的客观指标，
    与 `players/mock.py` 的 MockJudge 同源，故「客观值」口径唯一。
    """

    def __init__(self, profile: JudgeProfile,
                 objective_fn: Optional[callable] = None):
        self.profile = profile
        self.id = profile.judge_id
        self.model = f"biased:{profile.judge_id}"
        self._rng = random.Random(profile.seed)
        # 注入目标：便于测试用固定客观值，不必构造完整 Puzzle
        self._objective = objective_fn

    # -- Judge 契约 ----------------------------------------------------
    def score(self, puzzle, guess: str):
        if self._objective is not None:
            objective = self._objective(puzzle, guess)
        else:
            from engine.evaluate import char_f1
            objective = char_f1(guess, puzzle.solution) * 100.0

        p = self.profile
        raw = objective * p.slope + p.intercept
        if p.sigma > 0:
            raw += self._rng.gauss(0.0, p.sigma)
        value = max(SCORE_MIN, min(SCORE_MAX, raw))
        rationale = (f"客观基准 {objective:.1f} → 尺度×{p.slope:.2f} "
                     f"截距{p.intercept:+.1f} 噪声σ={p.sigma:.1f}")
        return round(value, 2), rationale

    # -- 受控复现 ------------------------------------------------------
    def reset_rng(self) -> None:
        """把噪声发生器复位到初始状态，使同一裁判可被重复评估。"""
        self._rng = random.Random(self.profile.seed)

    def with_profile(self, **kw) -> "BiasedJudge":
        return BiasedJudge(replace(self.profile, **kw), self._objective)


def make_homogeneous_panel(n: int = 4, sigma: float = 8.0,
                           seed0: int = 0) -> list:
    """同质裁判组：所有人误差相同 → R 应为 1.0，预期加权无增益。

    这是**零效应对照组**，用于验证「加权共识并非无条件更好」。
    """
    return [BiasedJudge(JudgeProfile(f"j{i}", slope=1.0, intercept=0.0,
                                     sigma=sigma, seed=seed0 + i))
            for i in range(n)]


def make_heterogeneous_panel(n: int = 4, seed0: int = 0) -> list:
    """异质裁判组：可靠性呈阶梯（噪声 2 / 8 / 20 / 35），其中一人还有自偏好。

    对应真实场景：裁判水平本就参差不齐，且存在系统性自偏好。
    """
    sigmas = [2.0, 8.0, 20.0, 35.0]
    intercepts = [0.0, 0.0, 0.0, 12.0]  # 最差的那个还自带 +12 自偏好
    out = []
    for i in range(n):
        out.append(BiasedJudge(JudgeProfile(
            f"j{i}", slope=1.0,
            intercept=intercepts[i % len(intercepts)],
            sigma=sigmas[i % len(sigmas)],
            seed=seed0 + i,
        )))
    return out
