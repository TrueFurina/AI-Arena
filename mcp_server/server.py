"""AI 角斗场 MCP 服务器入口。

暴露三类能力：
- Tools：开局 / 推演 / 跑通 / 列题 / 评测 / 报告
- Resources：题库、对局日志、模型注册表（可被客户端订阅）
- Prompts：玩家回合、裁判应答、赛后复盘（与引擎内部 prompt 共用一套模板）

传输：stdio（默认）。运行：python run.py 或 arena-mcp。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_server import config
from mcp_server.engine import game, puzzles
from mcp_server.engine.prompts import build_judge_prompt
from mcp_server.eval import metrics, report

mcp = FastMCP("ai-arena")

REPORT_DIR = config.ROOT / "reports"


# ============================== Tools ==============================
@mcp.tool()
def list_puzzles(difficulty: str | None = None, tags: list[str] | None = None,
                 limit: int | None = None) -> list[dict[str, Any]]:
    """列出题库（不含汤底，避免向玩家泄漏答案）。可按难度/标签过滤。"""
    return puzzles.list_puzzles(difficulty=difficulty, tags=tags, limit=limit)


@mcp.tool()
def get_puzzle(puzzle_id: str, include_truth: bool = False) -> dict[str, Any]:
    """按 id 取题。include_truth 默认 False（仅脱敏元数据）。

    ⚠️ 安全约定：玩家智能体禁止以 include_truth=true 调用本工具，
    否则会泄漏答案，破坏评测公平性。仅裁判/引擎内部使用真相。
    """
    p = puzzles.get_puzzle(puzzle_id, include_truth=include_truth)
    if p is None:
        return {"error": f"不存在 puzzle_id={puzzle_id}"}
    return p


@mcp.tool()
def start_match(puzzle_id: str, player_models: list[str], max_rounds: int | None = None,
                mode: str = "turtle_soup_qa") -> dict[str, Any]:
    """开一局海龟汤对战。

    - puzzle_id：题目 id（先用 list_puzzles 查）
    - player_models：玩家模型列表，如 ["mock","mock"]（离线）或 ["gpt-4o-mini","deepseek-chat"]
    - max_rounds：最大回合数（默认 20）
    - mode：对局模式（v0.1 仅 turtle_soup_qa）
    返回 match_id，随后用 step_match / run_match 推进。
    """
    try:
        mid = game.create_match(puzzle_id, player_models, max_rounds=max_rounds, mode=mode)
    except ValueError as e:
        return {"error": str(e)}
    return {"match_id": mid, "message": "对局已创建，可用 run_match 一键跑通或 step_match 逐步推演"}


@mcp.tool()
def get_match_state(match_id: str) -> dict[str, Any]:
    """获取某局当前状态：阶段、回合、对话记录、已揭示事实、比分。"""
    s = game.get_state(match_id)
    if s is None:
        return {"error": f"不存在 match_id={match_id}"}
    return s


@mcp.tool()
def step_match(match_id: str) -> dict[str, Any]:
    """推进一步（一名玩家行动一次），返回推进后的状态。用于逐步观察/中途介入。"""
    try:
        return game.step(match_id)
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
def run_match(match_id: str) -> dict[str, Any]:
    """非交互跑完整局（自动推进到终态），返回结算。适合批量自动评测。"""
    try:
        return game.run(match_id)
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
def list_matches(limit: int | None = None, status: str | None = None) -> list[dict[str, Any]]:
    """列出历史对局（按时间倒序）。status 可过滤 playing/finished。"""
    return game.list_matches(limit=limit, status=status)


@mcp.tool()
def analyze_match(match_id: str) -> dict[str, Any]:
    """对单局做基础能力分析：每位玩家的提问数、最佳猜想命中率、是否获胜。"""
    s = game.get_state(match_id)
    if s is None:
        # 尝试从磁盘加载
        from mcp_server.store import logs
        s = logs.load_match(match_id)
    if s is None:
        return {"error": f"不存在 match_id={match_id}"}
    return {"match_id": match_id, "metrics": metrics.match_metrics(s)}


@mcp.tool()
def get_leaderboard() -> dict[str, Any]:
    """跨所有对局聚合各模型表现：对局数、胜场、胜率、平均最佳猜想命中率。"""
    board = metrics.compute_leaderboard()
    return {"leaderboard": board.to_dict("records")}


@mcp.tool()
def export_report() -> dict[str, Any]:
    """生成评测报告：柱状图 PNG + 文字结论 MD，返回文件路径与结论文本。"""
    result = report.export_report(REPORT_DIR)
    return result


# ============================== Resources ==============================
@mcp.resource("arena://puzzles")
def res_puzzles() -> str:
    """题库（脱敏元数据，不含汤底）。"""
    return json.dumps(puzzles.list_puzzles(), ensure_ascii=False, indent=2)


@mcp.resource("arena://matches/{match_id}")
def res_match(match_id: str) -> str:
    """单局对局日志（JSON）。"""
    from mcp_server.store import logs
    s = logs.load_match(match_id) or game.get_state(match_id)
    if s is None:
        return json.dumps({"error": f"不存在 match_id={match_id}"}, ensure_ascii=False)
    return json.dumps(s, ensure_ascii=False, indent=2)


@mcp.resource("arena://models")
def res_models() -> str:
    """已配置模型注册表快照（不含任何 API key）。"""
    return json.dumps(config.list_models(), ensure_ascii=False, indent=2)


# ============================== Prompts ==============================
@mcp.prompt()
def player_turn(match_id: str) -> str:
    """生成某局当前应行动玩家的回合提示（与引擎内部 prompt 一致）。"""
    try:
        return game.build_player_turn_prompt(match_id)
    except ValueError as e:
        return str(e)


@mcp.prompt()
def judge_answer(question: str, puzzle_id: str) -> str:
    """生成汤主（裁判）对某个问题的 yes/no 应答提示。"""
    p = puzzles.get_puzzle(puzzle_id, include_truth=True)
    if p is None:
        return f"不存在 puzzle_id={puzzle_id}"
    return build_judge_prompt(question, p)


@mcp.prompt()
def debrief(match_id: str) -> str:
    """生成赛后复盘 / 真相揭示提示。"""
    try:
        return game.get_debrief(match_id)
    except ValueError as e:
        return str(e)


def main() -> None:
    """stdio 入口。"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
