"""评测指标与矩阵析因（全项目口径单一真值）。

为什么单独成模块：
  指标口径一旦漂移，历史结论全部失效——B 方案（run_compare.py）的
  DD=0.53 / GG=0.29、cross-play 2×2 析因「提问者效应主导」，都依赖
  这里的函数。此前这些实现散落在 run_crossplay.py 与 players/mock.py
  （三份拷贝），任一份改动都会造成数字打架。本模块收敛为唯一真值，
  并由 tests/test_evaluate.py 用历史日志锁定（变异即 FAIL）。

指标：
- f1     字符级 F1（客观主指标）：不依赖 LLM 裁判，规避裁判自偏好偏差
- vra    有效提问率：提问以问号结尾或含疑问词的比例
- yn     主持 yes/no 回答率

已知局限（诚实，勿夸大）：
  字符级 F1 只看字符集合、不考虑语序与语义，对长文本还原偏乐观，
  仅作相对排序用，不可当绝对还原度。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

QUEST_WORDS = ("吗", "呢", "为何", "为什么", "怎么", "是否", "是不是", "是谁",
               "什么", "哪里", "哪", "多少", "几")
YN_TOKENS = ("是", "否", "对", "不对", "正确", "错误", "yes", "no", "Yes", "No", "YES", "NO")

METRIC_KEYS = ("vra", "yn_rate", "f1")


def char_f1(a: str, b: str) -> float:
    """字符级 F1：字符集合的精确率/召回率调和平均（对称、有界 [0,1]）。"""
    sa, sb = set(a or ""), set(b or "")
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    if not inter:
        return 0.0
    prec = len(inter) / len(sa)
    rec = len(inter) / len(sb)
    return 2 * prec * rec / (prec + rec)


def is_valid_question(q: str) -> bool:
    """有效提问：以问号结尾，或含中文疑问词。"""
    q = (q or "").strip()
    if not q:
        return False
    if q.endswith(("?", "？")):
        return True
    return any(w in q for w in QUEST_WORDS)


def yn_rate(answers: Sequence[str]) -> float:
    """主持回答落在 yes/no 类词表中的比例。"""
    if not answers:
        return 0.0
    hit = sum(1 for m in answers if any(t in (m or "") for t in YN_TOKENS))
    return hit / len(answers)


def metrics_from_result(res: dict) -> dict:
    """从 GameResult dict 提取指标（提问 / 主持 / 还原 分离）。

    口径：与 B 方案、cross-play 实验完全一致，任何改动都必须同步
    tests/test_evaluate.py 的历史锁定值。
    """
    tr = res.get("transcript", [])
    qs = [m["content"] for m in tr if m.get("role") == "question"]
    ans = [m["content"] for m in tr if m.get("role") == "answer"]
    guess = next((m["content"] for m in tr if m.get("role") == "guess"), "")
    solution = res.get("puzzle", {}).get("solution", "")
    vq = sum(1 for x in qs if is_valid_question(x))
    return {
        "valid_q": vq,
        "total_q": len(qs),
        "vra": vq / max(1, len(qs)),
        "yn_rate": yn_rate(ans),
        "f1": char_f1(guess, solution),
        "guess": guess[:200],
    }


class Evaluator:
    """N×M 评测矩阵（行=提问者，列=主持）+ 2×2 析因。

    把 cross-play 的一次性 2×2 计算泛化为可复用结构，供后续
    「LLM × RL 策略」混合矩阵评测（L2 对齐）直接复用。
    """

    def __init__(self) -> None:
        self._cells: Dict[tuple, List[dict]] = {}

    def add(self, row: str, col: str, records: Sequence[dict]) -> None:
        self._cells.setdefault((row, col), []).extend(records)

    def add_result(self, row: str, col: str, res: dict) -> None:
        """直接喂 GameResult dict（走统一口径）。"""
        self.add(row, col, [metrics_from_result(res)])

    def cell(self, row: str, col: str) -> List[dict]:
        return self._cells.get((row, col), [])

    def agg(self, row: str, col: str) -> Dict[str, float]:
        rows = self.cell(row, col)
        if not rows:
            raise KeyError(f"空单元: ({row}, {col})")
        out: Dict[str, float] = {"n": float(len(rows))}
        for k in METRIC_KEYS:
            out[k] = sum(r[k] for r in rows) / len(rows)
        return out

    def rows(self) -> List[str]:
        return sorted({r for r, _ in self._cells})

    def cols(self) -> List[str]:
        return sorted({c for _, c in self._cells})

    def matrix(self, metric: str, rows: Sequence[str], cols: Sequence[str]) -> List[List[float]]:
        return [[self.agg(r, c)[metric] for c in cols] for r in rows]

    def factorial_2x2(self, r0: str, r1: str, c0: str, c1: str,
                      metric: str = "f1") -> dict:
        """2×2 析因：主效应 + 交互 + 可加性检验。

        r0/r1 = 行的两种取值（如两种提问者），c0/c1 = 列的两种取值（两种主持）。
        - row_effect_at_c*：固定列，换行的效应（提问者效应）
        - col_effect_at_r*：固定行，换列的效应（主持效应）
        - interaction：两主效应之差，衡量协同/拮抗
        - additive_gap：|对角差 - 交叉差|，越小说明角色效应越接近可加
        """
        f00 = self.agg(r0, c0)[metric]
        f01 = self.agg(r0, c1)[metric]
        f10 = self.agg(r1, c0)[metric]
        f11 = self.agg(r1, c1)[metric]

        row_eff_c0 = f00 - f10
        row_eff_c1 = f01 - f11
        col_eff_r0 = f00 - f01
        col_eff_r1 = f10 - f11
        interaction = row_eff_c0 - row_eff_c1
        diag_diff = f00 - f11
        cross_diff = f01 - f10

        row_mag = abs(row_eff_c0) + abs(row_eff_c1)
        col_mag = abs(col_eff_r0) + abs(col_eff_r1)
        return {
            "metric": metric,
            "cells": {f"({r0},{c0})": f00, f"({r0},{c1})": f01,
                      f"({r1},{c0})": f10, f"({r1},{c1})": f11},
            "row_effect_at_c0": row_eff_c0,
            "row_effect_at_c1": row_eff_c1,
            "col_effect_at_r0": col_eff_r0,
            "col_effect_at_r1": col_eff_r1,
            "interaction": interaction,
            "diag_diff": diag_diff,
            "cross_diff": cross_diff,
            "additive_gap": abs(diag_diff - cross_diff),
            "dominant": "row" if row_mag >= col_mag else "col",
        }
