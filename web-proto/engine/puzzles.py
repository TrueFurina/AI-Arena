"""题库加载：从 puzzles.json 读取海龟汤题目。"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from .game import Puzzle

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PATH = os.path.join(os.path.dirname(_HERE), "puzzles.json")


def load_puzzles(path: str = _DEFAULT_PATH) -> List[Puzzle]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Puzzle(**p) for p in raw]


def get_puzzle(puzzle_id: Optional[str] = None, path: str = _DEFAULT_PATH) -> Puzzle:
    puzzles = load_puzzles(path)
    if not puzzles:
        raise ValueError("题库为空")
    if puzzle_id is None:
        return puzzles[0]
    for p in puzzles:
        if p.soup_id == puzzle_id:
            return p
    raise KeyError(f"找不到题目: {puzzle_id}")


if __name__ == "__main__":
    for p in load_puzzles():
        print(p.soup_id, "|", p.title)
