"""服务器测试：直接调用工具函数验证集成，并用 FastMCP 内存客户端做 MCP 层端到端验证。"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server import server as srv  # noqa: E402


def _unwrap(res):
    """FastMCP 3.x：dict 返回直接在 res.data，list 返回包在 Output.result 里。"""
    d = res.data
    return d.result if hasattr(d, "result") else d


def _resource_text(mr):
    """资源返回可能是 content 列表或带 .result 的对象。"""
    if isinstance(mr, list):
        first = mr[0]
        return first.content if hasattr(first, "content") else first.text
    d = mr.data
    return d.result if hasattr(d, "result") else d


def test_tools_direct_integration():
    """直接调用工具函数，验证 开局→跑通→分析→排行榜 全链路。"""
    created = srv.start_match("ts-001", ["mock", "mock"], max_rounds=20)
    assert "match_id" in created
    mid = created["match_id"]

    final = srv.run_match(mid)
    assert final["status"] == "finished"
    assert final["winner"] is not None

    analysis = srv.analyze_match(mid)
    assert "metrics" in analysis
    assert analysis["metrics"][final["winner"]]["win"] is True

    # 默认排行榜剔除作弊玩家（mock 预言机掌握真相，成绩不反映真实能力）
    board = srv.get_leaderboard()
    assert "mock" not in {r["model"] for r in board["leaderboard"]}

    # 显式要求时才包含
    board_all = srv.get_leaderboard(include_cheating=True)
    assert "mock" in {r["model"] for r in board_all["leaderboard"]}


def test_list_puzzles_no_truth():
    out = srv.list_puzzles()
    assert len(out) >= 1
    assert all("truth" not in p for p in out)


def test_mcp_client_in_memory():
    """用 FastMCP 内存客户端验证 MCP 注册与传输真正可用（工具/资源/提示）。"""
    from fastmcp import Client

    async def go():
        async with Client(srv.mcp) as client:
            tools = await client.list_tools()
            assert "list_puzzles" in {t.name for t in tools}

            lp = _unwrap(await client.call_tool("list_puzzles", {}))
            assert len(lp) >= 1 and all("truth" not in p for p in lp)

            mid = _unwrap(
                await client.call_tool(
                    "start_match",
                    {"puzzle_id": "ts-001", "player_models": ["mock", "mock"], "max_rounds": 20},
                )
            )["match_id"]

            st = _unwrap(await client.call_tool("run_match", {"match_id": mid}))
            assert st["status"] == "finished"
            assert st["winner"] is not None

            an = _unwrap(await client.call_tool("analyze_match", {"match_id": mid}))
            assert an["metrics"][st["winner"]]["win"] is True

            lb = _unwrap(await client.call_tool("get_leaderboard", {}))
            assert isinstance(lb["leaderboard"], list)

            rep = _unwrap(await client.call_tool("export_report", {}))
            assert rep["text"].startswith("#")

            mr = await client.read_resource("arena://matches/" + mid)
            assert "match_id" in json.loads(_resource_text(mr))

            pr = await client.get_prompt("player_turn", {"match_id": mid})
            assert len(pr.messages[0].content.text) > 0

    asyncio.run(go())
