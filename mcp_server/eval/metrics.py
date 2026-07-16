"""评测指标（v0.1 基础版）。

v0.1 聚焦可量化、可解释的基础指标；v0.3 会深化为
推理 / 欺骗 / 伪装 / 辩论 四维能力评分。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from mcp_server.store import logs


def match_metrics(state: dict[str, Any]) -> dict[str, Any]:
    """从单局状态计算每位玩家的指标。"""
    out = {}
    winner = state.get("winner")
    for m, sc in state.get("scores", {}).items():
        out[m] = {
            "questions_asked": sc.get("questions", 0),
            "best_guess": round(sc.get("best_guess", 0.0), 3),
            "win": bool(sc.get("win", False)),
            "is_winner": m == winner,
        }
    return out


def compute_leaderboard() -> pd.DataFrame:
    """跨所有已存储对局，聚合各模型的胜场、对局数、平均最佳猜想命中率。"""
    rows = []
    for meta in logs.list_match_metas():
        s = logs.load_match(meta["match_id"])
        if not s:
            continue
        for m, sc in s.get("scores", {}).items():
            rows.append(
                {
                    "model": m,
                    "match_id": s["match_id"],
                    "questions_asked": sc.get("questions", 0),
                    "best_guess": sc.get("best_guess", 0.0),
                    "win": 1 if sc.get("win") else 0,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["model", "games", "wins", "win_rate", "avg_best_guess"])

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("model")
        .agg(games=("match_id", "count"), wins=("win", "sum"),
             avg_best_guess=("best_guess", "mean"))
        .reset_index()
    )
    agg["win_rate"] = (agg["wins"] / agg["games"]).round(3)
    agg["avg_best_guess"] = agg["avg_best_guess"].round(3)
    return agg.sort_values(["wins", "avg_best_guess"], ascending=False).reset_index(drop=True)
