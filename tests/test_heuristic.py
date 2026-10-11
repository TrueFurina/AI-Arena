"""启发式玩家测试：验证"不作弊"与"有区分度"这两个核心性质。

v0.1 的 MockPlayer 是掌握真相的预言机，两个 mock 对战必定打平且都满分，
排行榜因此毫无意义。HeuristicPlayer 必须解决这两点：
1. 不偷看 truth / key_facts（评测才可信）
2. 不同性格策略产生不同得分（排行榜才有意义）
"""
from mcp_server.config import list_models
from mcp_server.engine import game
from mcp_server.engine.player import HeuristicPlayer, get_player
from mcp_server.engine.puzzles import get_puzzle


def test_heuristic_player_never_sees_truth():
    """启发式玩家只能拿到汤面，不得持有真相或关键事实。"""
    puzzle = get_puzzle("ts-001", include_truth=True)
    p = get_player("heuristic-conservative", puzzle)

    assert isinstance(p, HeuristicPlayer)
    assert p.uses_truth is False
    # 实例属性中不得出现真相字段
    assert "truth" not in p.__dict__
    assert "key_facts" not in p.__dict__
    assert "facts" not in p.__dict__
    # 唯一可见的题目信息就是汤面
    assert p.surface == puzzle["surface"]


def test_mock_player_is_flagged_as_cheating():
    """对照：mock 玩家明确标记为使用真相，便于排行榜排除。"""
    puzzle = get_puzzle("ts-001", include_truth=True)
    assert get_player("mock", puzzle).uses_truth is True


def test_heuristic_match_reaches_finished():
    """启发式玩家能把一局跑到终态，不会卡死。"""
    mid = game.create_match("ts-001", ["heuristic", "heuristic-conservative"], max_rounds=20)
    state = game.run(mid)
    assert state["status"] == "finished"
    assert state["transcript"], "对局未产生任何记录"


def test_heuristic_player_hits_some_facts():
    """虽不知道答案，靠汤面词 + 常识假设也应命中部分关键事实（否则得分恒为 0）。"""
    mid = game.create_match("ts-001", ["heuristic-conservative"], max_rounds=24)
    state = game.run(mid)
    assert state["revealed_facts"], "启发式玩家提问一个关键事实都没命中"


def test_heuristic_styles_yield_different_scores():
    """不同性格策略必须产生不同得分——这是排行榜有意义的前提。"""
    scores = {}
    for style in ("aggressive", "conservative", "balanced"):
        name = f"heuristic-{style}"
        mid = game.create_match("ts-003", [name], max_rounds=24)
        state = game.run(mid)
        scores[style] = state["scores"][name]["best_guess"]

    assert len(set(scores.values())) > 1, (
        f"各性格得分完全相同，排行榜将失去区分度: {scores}"
    )


def test_offline_players_are_listed():
    """离线玩家应对 MCP 客户端可见（arena://models），且无需 API key。"""
    models = {m["name"]: m for m in list_models()}
    for name in ("mock", "heuristic", "heuristic-aggressive", "heuristic-creative"):
        assert name in models, f"{name} 未出现在 arena://models"
        assert models[name]["offline"] is True
        assert models[name]["configured"] is True, "离线玩家不应依赖 API key"
