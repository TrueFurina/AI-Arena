"""对局日志存储：JSON 落盘到 data/matches/。

每次对局结束（或按需）持久化为 {match_id}.json，便于复盘、评测与资源订阅。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp_server.config import MATCHES_DIR


def _path(match_id: str) -> Path:
    return MATCHES_DIR / f"{match_id}.json"


def save_match(state: dict[str, Any]) -> None:
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(state["match_id"]), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_match(match_id: str) -> dict[str, Any] | None:
    p = _path(match_id)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def list_match_metas(limit: int | None = None, status: str | None = None) -> list[dict[str, Any]]:
    """列出对局元数据（不含完整 transcript，控制体积）。"""
    metas: list[dict[str, Any]] = []
    if not MATCHES_DIR.exists():
        return metas
    for p in sorted(MATCHES_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
        if status and s.get("status") != status:
            continue
        metas.append(
            {
                "match_id": s["match_id"],
                "puzzle_id": s.get("puzzle_id"),
                "mode": s.get("mode"),
                "status": s.get("status"),
                "round": s.get("round"),
                "players": s.get("players"),
                "winner": s.get("winner"),
                "created_at": s.get("created_at"),
            }
        )
    if limit is not None:
        metas = metas[:limit]
    return metas
