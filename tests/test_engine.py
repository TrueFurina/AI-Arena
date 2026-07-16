"""引擎测试：Mock 玩家驱动一局到达终态，验证胜负与日志结构。"""
from mcp_server.engine import game


def test_mock_match_reaches_finished():
    mid = game.create_match("ts-001", ["mock", "mock"], max_rounds=20)
    state = game.run(mid)
    assert state["status"] == "finished"
    assert state["winner"] is not None
    # 胜者是两名 mock 之一
    assert state["winner"] in state["players"]


def test_initial_state_no_truth_leak():
    mid = game.create_match("ts-002", ["mock"], max_rounds=5)
    state = game.get_state(mid)
    assert "_puzzle" not in state
    assert "_players" not in state
    assert "truth" not in state


def test_step_records_transcript():
    mid = game.create_match("ts-003", ["mock"], max_rounds=10)
    game.step(mid)
    state = game.get_state(mid)
    assert len(state["transcript"]) == 1
    rec = state["transcript"][0]
    assert rec["action"] in ("ask", "guess")
    if rec["action"] == "ask":
        assert rec["host_answer"] in ("yes", "no")


def test_list_matches_includes_created():
    mid = game.create_match("ts-004", ["mock"], max_rounds=8)
    game.run(mid)
    metas = game.list_matches()
    ids = [m["match_id"] for m in metas]
    assert mid in ids
