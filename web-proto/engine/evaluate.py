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
- dhit   去重线索命中率：不同问句中被确认的数量 / 提问总轮数
- tdhit  真值感知去重命中率（P2-B 欺骗环境主指标，见下）

已知局限（诚实，勿夸大）：
  字符级 F1 只看字符集合、不考虑语序与语义，对长文本还原偏乐观，
  仅作相对排序用，不可当绝对还原度。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

QUEST_WORDS = ("吗", "呢", "为何", "为什么", "怎么", "是否", "是不是", "是谁",
               "什么", "哪里", "哪", "多少", "几")
YN_TOKENS = ("是", "否", "对", "不对", "正确", "错误", "yes", "no", "Yes", "No", "YES", "NO")
CONFIRM_ANSWERS = ("是", "对", "正确", "是的", "没错", "yes")

METRIC_KEYS = ("vra", "yn_rate", "f1", "hit_rate", "dhit")


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


def is_confirm(answer: str) -> bool:
    """主持回答是否构成「确认」（口径唯一真值，rl.py 亦从此处导入）。

    单独收敛的理由：终局还原只采纳被确认的线索，而 RL 奖励同样依赖该
    判定。若两处各有一份，判定漂移会让「F1 通路」的结论悄悄失效。
    """
    a = (answer or "").strip()
    return a in CONFIRM_ANSWERS or a.startswith("是")


def hit_rate(answers: Sequence[str]) -> float:
    """线索命中率：回答被确认为「是」的比例。

    为什么在 F1 之外还要这个指标：F1 会被汤底措辞稀释——提问者问中了
    「身高」，主持答「是」，但汤底写的是「个子矮」，字符不重合，F1 不涨。
    hit_rate 不受措辞影响，是**提问质量**的直接度量；F1 是**还原质量**
    的度量。两者要分开报，混用会得出「提问没进步」的错误结论。
    """
    if not answers:
        return 0.0
    return sum(1 for a in answers if is_confirm(a)) / len(answers)


def distinct_hit_rate(transcript: Sequence[dict]) -> float:
    """去重线索命中率 = 不同问句中被确认的数量 / **提问总轮数**。

    为什么需要它（语义化动作引入后必须补）：
      hit_rate 完全不惩罚重复提问。关键词动作用时问题不大——同一关键词在
      新题上通常答「无关」，重复问拿不到分。但**语义类别是跨题共享**的：
      某类别在这题上答「是」，重复问 8 次就拿满 hit_rate=1.0，指标被刷爆。

    ⚠️ 分母必须是**提问总轮数**，不能是「不同问句数」——我第一版就是按
    不同问句数做分母，结果重复提问既不增分子也不增分母，重复问一个命中
    的问题照样得 1.0，等于没堵住漏洞。分母取总轮数后：
        重复问同一个命中问题 8 次  → 1/8 = 0.125（只探到 1 条线索）
        问 8 个不同问题命中 5 条   → 5/8 = 0.625
    这才符合「收集到的信息量」的直觉。
    """
    pairs: Dict[str, str] = {}
    pending: Optional[str] = None
    n_questions = 0
    for m in transcript:
        role = m.get("role")
        if role == "question":
            pending = m.get("content", "")
            n_questions += 1
        elif role == "answer" and pending is not None:
            pairs.setdefault(pending, m.get("content", ""))
            pending = None
    if not n_questions:
        return 0.0
    return sum(1 for a in pairs.values() if is_confirm(a)) / n_questions


def tdhit(transcript: Sequence[dict], truth_fn) -> float:
    """真值感知去重命中率（P2-B 欺骗环境主指标，DESIGN-P2B-MARL.md §3.5）。

    tdhit = 不同问句中答案与如实答案一致的数量 / 提问总轮数

    为什么 dhit 不够：对说谎主持，dhit 可被谎言刷高——主持见「否」
    即「是」，dhit 照单全收（表观命中照涨），而 F1 被假线索打穿。
    **用 dhit 评欺骗环境会得出反向结论**。tdhit 只计与如实答案一致的
    回答，谎言（注入假阳性或漏报真线索）都计不到。

    口径细节（与 distinct_hit_rate 对齐，防重复刷分）：
    - 同一问句多次出现只取其**首次**答案（setdefault 语义）；
    - 分母是**提问总轮数**（不是不同问句数），复问稀释得分；
    - truth_fn(question) 返回 None（问题无法解析）→ 不计入分子，
      但仍占分母——「收不到可验证信息的轮次」本身就是信息缺失。

    边界：truth_fn 由调用方注入（MARL 层用 category_answer），engine 不
    耦合语义模块；答案比较为**去空格精确匹配**——当前主持（CategoryHost/
    DeceptiveHost）只产 是/否。若将来接入措辞多变的 LLM 主持，先定归一化
    口径并补测试，不得 silently 放宽。刻意**不进** METRIC_KEYS：
    Evaluator.agg 按固定键聚合历史记录，加键会让旧记录 KeyError。
    """
    pairs: Dict[str, str] = {}
    pending: Optional[str] = None
    n_questions = 0
    for m in transcript:
        role = m.get("role")
        if role == "question":
            pending = m.get("content", "")
            n_questions += 1
        elif role == "answer" and pending is not None:
            pairs.setdefault(pending, m.get("content", ""))
            pending = None
    if not n_questions:
        return 0.0
    hit = 0
    for q, a in pairs.items():
        t = truth_fn(q)
        if t is not None and (a or "").strip() == (t or "").strip():
            hit += 1
    return hit / n_questions


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
        "hit_rate": hit_rate(ans),
        "dhit": distinct_hit_rate(tr),
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
