"""题库加载与查询。

题库以 JSON 形式存放在 data/puzzles.json。本模块负责加载、按条件过滤，
并提供"脱敏元数据"（不含汤底）与"完整题目"（含汤底，仅裁判/引擎使用）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp_server.config import DATA_DIR

PUZZLES_FILE = DATA_DIR / "puzzles.json"


def _load_all() -> list[dict[str, Any]]:
    with open(PUZZLES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def list_puzzles(difficulty: str | None = None, tags: list[str] | None = None,
                 limit: int | None = None) -> list[dict[str, Any]]:
    """返回题库脱敏元数据（不含 truth / key_facts 细节用于泄露控制，但保留 tags 以便筛选）。"""
    puzzles = _load_all()
    if difficulty:
        puzzles = [p for p in puzzles if p.get("difficulty") == difficulty]
    if tags:
        wanted = set(tags)
        puzzles = [p for p in puzzles if wanted.intersection(set(p.get("tags", [])))]
    if limit is not None:
        puzzles = puzzles[:limit]
    # 脱敏：不返回 truth
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "difficulty": p["difficulty"],
            "tags": p.get("tags", []),
            "surface": p["surface"],
        }
        for p in puzzles
    ]


def get_puzzle(puzzle_id: str, include_truth: bool = False) -> dict[str, Any] | None:
    """按 id 取题。include_truth=False 时剔除 truth / key_facts，避免向玩家泄漏答案。"""
    for p in _load_all():
        if p["id"] == puzzle_id:
            if include_truth:
                return p
            return {
                "id": p["id"],
                "title": p["title"],
                "difficulty": p["difficulty"],
                "tags": p.get("tags", []),
                "surface": p["surface"],
            }
    return None


def all_puzzle_ids() -> list[str]:
    return [p["id"] for p in _load_all()]
