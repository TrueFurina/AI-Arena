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

    # deception 模式：首位玩家是知情者（掌握真相并可撒谎），其余为求解者。
    # 回合只在求解者之间轮转，知情者只负责应答。
    insider: str | None = None
    solvers: list[str] = list(player_models)
    if mode == "deception":
        if len(player_models) < 2:
            raise ValueError("deception 模式至少需要 1 名知情者 + 1 名求解者")
        insider = player_models[0]
        solvers = list(player_models[1:])

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
        # deception 模式下的角色划分；普通模式 insider=None、solvers=players
        "insider": insider,
        "solvers": solvers,
        "transcript": [],
        "revealed_facts": [],
        # 本轮内已提交但尚未结算的猜想 {player: score}，轮末统一比较
        "pending_guesses": {},
        # deception 模式：质疑统计（公开，质疑对错本身是游戏规则的一部分）
        "challenges": {"total": 0, "success": 0},
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

    # deception 模式下回合只在求解者之间轮转（知情者只应答，不占回合）
    players = s.get("solvers") or s["players"]
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

    if record["action"] == "challenge":
        # 指认"你在撒谎"：核验玩家指出的那个问题是否确实被撒过谎。
        # 前后回答不一致 ⇒ 至少有一次是谎，故只要该问题存在撒谎记录即成立。
        target = action.get("target", "")
        log = s.get("_lie_log", [])
        lied = any(r.get("lied") and r.get("question") == target for r in log)
        record["challenge_success"] = lied
        record["target"] = target
        s["challenges"]["total"] += 1
        if lied:
            s["challenges"]["success"] += 1
    elif record["action"] in ("guess", "argue"):
        # argue 是 debate 模式的收网动作：提交一段论证，与对手比较说服力
        correct, score = evaluate_guess(record["content"], puzzle, GUESS_THRESHOLD)
        record["score"] = round(score, 3)
        record["correct"] = correct
        s["scores"][current]["best_guess"] = max(s["scores"][current]["best_guess"], score)
        # 不在此处立即定胜负：先记入本轮待结算，等所有玩家都行动完再统一比较，
        # 否则先手玩家会抢先终结对局，后手连提交猜想的机会都没有。
        s["pending_guesses"][current] = score
    else:
        # 提问：先由裁判算出真实答案（ground truth）
        truth_answer, matched = answer_question(record["content"], puzzle)

        if s.get("insider"):
            # deception 模式：由知情者对外应答，他可以选择撒谎误导求解者
            insider_obj = s["_players"][s["insider"]]
            answer, lied = insider_obj.answer(record["content"], truth_answer)
            record["host_answer"] = answer
            # 是否撒谎属于评测数据，绝不能放进玩家可见的 transcript
            s.setdefault("_lie_log", []).append(
                {
                    "round": s["round"],
                    "asker": current,
                    "question": record["content"],
                    "truth_answer": truth_answer,
                    "given_answer": answer,
                    "lied": lied,
                }
            )
            if lied:
                matched = None  # 撒了谎就不该揭示真实事实
        else:
            record["host_answer"] = truth_answer

        record["matched_fact"] = matched
        if matched and matched not in s["revealed_facts"]:
            s["revealed_facts"].append(matched)
        s["scores"][current]["questions"] += 1

    s["transcript"].append(record)
    s["turn_index"] += 1
    # 一轮 = 所有玩家各行动一次
    if s["turn_index"] % n == 0:
        s["round"] += 1
        # 轮末统一结算本轮所有猜想，保证同轮竞争公平
        winner = _settle_round(s)
        if winner:
            _finish(s, winner=winner)
            _persist(s)
            return _public_state(s)

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


def _settle_round(s: dict[str, Any]) -> str | None:
    """轮末统一结算：比较本轮所有玩家提交的猜想，命中率最高且达标者胜。

    为什么不在玩家猜中的瞬间就结束对局：那样先手玩家会抢在所有人之前终结比赛，
    后手玩家连提交猜想的机会都没有——评测到的其实是"手速 + 出场顺序"，
    而非推理能力。统一结算后，同一轮内提交的猜想被同等对待，后手依然有完整的
    一轮时间作答，胜负只取决于猜想质量。

    平分不再"按顺序送胜"：分数并列时先到手的玩家会仅因出手顺序被宣布获胜，
    这是另一种先手优势。平分必须走加时赛（见 _break_tie），加时仍持平则
    返回 None——宁可比赛继续，也不能把胜利送给字典序。
    """
    pending = s.get("pending_guesses", {})
    if not pending:
        return None
    best_score = max(pending.values())
    leaders = [m for m, sc in pending.items() if sc == best_score]
    s["pending_guesses"] = {}
    # debate 是"比较制"：谁的解释更有说服力谁赢，不要求达到绝对阈值——
    # 这正是辩论对抗性的体现，赢的是相对更强的一方。
    if s.get("mode") == "debate":
        if best_score <= 0:
            return None
        return _break_tie(s, leaders, "evidence")
    if best_score >= GUESS_THRESHOLD:
        return _break_tie(s, leaders, "efficiency")
    return None


def _break_tie(s: dict[str, Any], leaders: list[str], axis: str) -> str | None:
    """平分加时赛：命中率相同时，按模式选一条能体现能力的副标准决胜。

    - ``evidence``（debate 比较制）：比论据充分度——各自提问中被裁判判为
      yes 的总数。论据越多且可验证，论证越有说服力。
    - ``efficiency``（qa 阈值制）：比提问效率——命中率相同的情况下，
      提问更少的人推理效率更高。

    副标准仍完全持平（双方行为逐帧一致）时返回 None：宁可让比赛继续
    轮次或最终无人胜出，也不能把胜利送给出手顺序——那评测的是运气，
    不是推理/辩论能力。
    """
    if len(leaders) == 1:
        return leaders[0]

    if axis == "evidence":
        def key(m: str) -> int:
            return sum(
                1 for r in s.get("transcript", [])
                if r.get("player") == m and r.get("action") == "ask"
                and r.get("host_answer") == "yes"
            )
    else:  # efficiency：同样命中率下提问越少越高效（取负号让 max 选中它）
        def key(m: str) -> int:
            return -s["scores"][m]["questions"]

    best = max(leaders, key=key)
    tied = [m for m in leaders if key(m) == key(best)]
    return None if len(tied) > 1 else best


def _finish(s: dict[str, Any], winner: str | None) -> None:
    s["status"] = "finished"
    s["phase"] = "finished"
    s["winner"] = winner
    if winner:
        s["scores"][winner]["win"] = True
    s["finished_at"] = time.time()


def _finish_by_best(s: dict[str, Any]) -> None:
    """达到最大回合数仍未分胜负：按最佳猜想命中率判定。

    与 _settle_round 同理——平分走加时赛，加时也持平则 winner=None
    （无人胜出），不按出手顺序送胜。
    """
    best_score = max(sc["best_guess"] for sc in s["scores"].values())
    if best_score <= 0:
        _finish(s, winner=None)
        return
    leaders = [m for m, sc in s["scores"].items() if sc["best_guess"] == best_score]
    axis = "evidence" if s.get("mode") == "debate" else "efficiency"
    _finish(s, winner=_break_tie(s, leaders, axis))


def _persist(s: dict[str, Any]) -> None:
    """落盘。玩家可见状态由 _public_state 剔除内部字段后保存。

    例外：`_lie_log`（知情者撒谎记录）是评测数据，必须落盘才能算欺骗指标；
    但它下划线开头，不会出现在 get_state 返回给玩家的状态里。
    """
    data = _public_state(s)
    if s.get("_lie_log"):
        data["_lie_log"] = s["_lie_log"]
    logs.save_match(data)


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
