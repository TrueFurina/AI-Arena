"""海龟汤游戏引擎：管理对局生命周期、回合推进与胜负结算。

状态模型 MatchState（dict 形式，便于 JSON 序列化与 MCP 资源暴露）：
- match_id, puzzle_id, mode, phase(playing|finished)
- round, max_rounds, turn_index
- players: list[str]（模型名）
- transcript: list[TurnRecord]
- revealed_facts: list[str]（汤主揭示过的关键事实）
- scores: {player: {best_guess, questions, win}}
- status, winner, created_at, finished_at
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from mcp_server.config import DEFAULT_MAX_ROUNDS, GUESS_THRESHOLD
from mcp_server.engine.judge import answer_question, debrief_text, evaluate_guess
from mcp_server.engine.player import get_player
from mcp_server.engine.prompts import build_player_prompt
from mcp_server.engine.puzzles import get_puzzle
from mcp_server.store import logs

# 内存中对局表（持久化由 logs 负责）
_MATCHES: dict[str, dict[str, Any]] = {}


def create_match(
    puzzle_id: str,
    player_models: list[str],
    max_rounds: int | None = None,
    mode: str = "turtle_soup_qa",
) -> str:
    puzzle = get_puzzle(puzzle_id, include_truth=True)
    if puzzle is None:
        raise ValueError(f"题库中不存在 puzzle_id={puzzle_id}")
    if not player_models:
        raise ValueError("至少需要一名玩家（player_models 不能为空）")

    match_id = f"match-{uuid.uuid4().hex[:8]}"
    state: dict[str, Any] = {
        "match_id": match_id,
        "puzzle_id": puzzle_id,
        "title": puzzle["title"],
        "surface": puzzle["surface"],
        "mode": mode,
        "phase": "playing",
        "round": 1,
        "turn_index": 0,
        "max_rounds": max_rounds or DEFAULT_MAX_ROUNDS,
        "players": list(player_models),
        "transcript": [],
        "revealed_facts": [],
        "scores": {
            m: {"best_guess": 0.0, "questions": 0, "win": False} for m in player_models
        },
        "status": "playing",
        "winner": None,
        "created_at": time.time(),
        "finished_at": None,
        "_puzzle": puzzle,  # 仅引擎内部使用，不对外暴露
        "_players": {m: get_player(m, puzzle) for m in player_models},
    }
    _MATCHES[match_id] = state
    return match_id


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    """去掉引擎内部字段，返回可对外暴露的状态。"""
    return {k: v for k, v in state.items() if not k.startswith("_")}


def get_state(match_id: str) -> dict[str, Any] | None:
    s = _MATCHES.get(match_id)
    return _public_state(s) if s else None


def step(match_id: str) -> dict[str, Any]:
    """推进一个回合（一名玩家行动一次）。"""
    s = _MATCHES.get(match_id)
    if s is None:
        raise ValueError(f"不存在 match_id={match_id}")
    if s["status"] != "playing":
        return _public_state(s)

    players = s["players"]
    n = len(players)
    current = players[s["turn_index"] % n]
    puzzle = s["_puzzle"]

    # 给玩家注入当前状态（含 surface / transcript / revealed_facts，不含 truth）
    view = _public_state(s)
    view["current_player"] = current
    player = s["_players"][current]
    action = player.act(view)

    record: dict[str, Any] = {
        "round": s["round"],
        "player": current,
        "action": action.get("action", "ask"),
        "content": action.get("content", ""),
    }

    if record["action"] == "guess":
        correct, score = evaluate_guess(record["content"], puzzle, GUESS_THRESHOLD)
        record["score"] = round(score, 3)
        record["correct"] = correct
        s["scores"][current]["best_guess"] = max(s["scores"][current]["best_guess"], score)
        if correct:
            _finish(s, winner=current)
            s["transcript"].append(record)
            _persist(s)
            return _public_state(s)
    else:
        # 提问：裁判应答并可能揭示关键事实
        answer, matched = answer_question(record["content"], puzzle)
        record["host_answer"] = answer
        record["matched_fact"] = matched
        if matched and matched not in s["revealed_facts"]:
            s["revealed_facts"].append(matched)
        s["scores"][current]["questions"] += 1

    s["transcript"].append(record)
    s["turn_index"] += 1
    # 一轮 = 所有玩家各行动一次
    if s["turn_index"] % n == 0:
        s["round"] += 1

    # 达到最大回合数仍未分胜负 → 按最佳猜想命中率判定
    if s["round"] > s["max_rounds"]:
        _finish_by_best(s)

    _persist(s)
    return _public_state(s)


def run(match_id: str) -> dict[str, Any]:
    """非交互跑完整局，直到终态。"""
    s = _MATCHES.get(match_id)
    if s is None:
        raise ValueError(f"不存在 match_id={match_id}")
    guard = 0
    while s["status"] == "playing" and guard < 10000:
        step(match_id)
        guard += 1
    return _public_state(s)


def _finish(s: dict[str, Any], winner: str | None) -> None:
    s["status"] = "finished"
    s["phase"] = "finished"
    s["winner"] = winner
    if winner:
        s["scores"][winner]["win"] = True
    s["finished_at"] = time.time()


def _finish_by_best(s: dict[str, Any]) -> None:
    best, best_score = None, -1.0
    for m, sc in s["scores"].items():
        if sc["best_guess"] > best_score:
            best, best_score = m, sc["best_guess"]
    _finish(s, winner=best if best_score > 0 else None)


def _persist(s: dict[str, Any]) -> None:
    """终态时落盘；进行中也可按需保存（这里在每步与终态都保存，便于中途查看）。"""
    logs.save_match(_public_state(s))


def list_matches(limit: int | None = None, status: str | None = None) -> list[dict[str, Any]]:
    return logs.list_match_metas(limit=limit, status=status)


def get_debrief(match_id: str) -> str:
    s = _MATCHES.get(match_id)
    if s is None:
        # 尝试从磁盘加载
        disk = logs.load_match(match_id)
        if disk is None:
            raise ValueError(f"不存在 match_id={match_id}")
        s = disk
    puzzle = get_puzzle(s["puzzle_id"], include_truth=True)
    return debrief_text(puzzle)


def build_player_turn_prompt(match_id: str) -> str:
    """为当前应行动的玩家生成 prompt（供 MCP prompt 资源复用）。"""
    s = _MATCHES.get(match_id)
    if s is None:
        raise ValueError(f"不存在 match_id={match_id}")
    players = s["players"]
    current = players[s["turn_index"] % len(players)]
    view = _public_state(s)
    view["current_player"] = current
    return build_player_prompt(view, current)
