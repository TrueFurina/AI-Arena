"""评测层测试：公平结算、多维指标、作弊排除、雷达图导出。"""
from pathlib import Path

from mcp_server.engine import game
from mcp_server.engine.game import _settle_round
from mcp_server.eval.metrics import compute_leaderboard, compute_profile, match_metrics
from mcp_server.eval.report import generate_radar


# --- 回合末统一结算（公平性） --------------------------------------------
def test_settle_round_picks_highest_above_threshold():
    s = {"pending_guesses": {"a": 0.3, "b": 0.7}}
    assert _settle_round(s) == "b"
    assert s["pending_guesses"] == {}, "结算后应清空待结算队列"


def test_settle_round_rejects_all_below_threshold():
    s = {"pending_guesses": {"a": 0.1, "b": 0.2}}
    assert _settle_round(s) is None, "无人达标时不应产生胜者"
    assert s["pending_guesses"] == {}


def test_settle_round_handles_empty():
    assert _settle_round({"pending_guesses": {}}) is None


def test_guess_does_not_end_match_before_others_act():
    """先手猜中不再立即终结对局：同一轮内其余玩家仍能行动。"""
    mid = game.create_match("ts-001", ["mock", "heuristic-conservative"], max_rounds=20)
    st = game.run(mid)

    correct_guesses = [
        r for r in st["transcript"]
        if r["player"] == "mock" and r["action"] == "guess" and r.get("correct")
    ]
    assert correct_guesses, "mock 应至少提交一次正确猜想"
    r0 = correct_guesses[0]
    same_round_others = [
        t for t in st["transcript"]
        if t["round"] == r0["round"] and t["player"] != "mock"
    ]
    assert same_round_others, "先手猜中后，同轮其他玩家没有行动记录（说明对局被提前结束）"


# --- 多维指标 ------------------------------------------------------------
def test_match_metrics_exposes_profile_dimensions():
    mid = game.create_match("ts-002", ["heuristic-conservative"], max_rounds=24)
    st = game.run(mid)
    mt = match_metrics(st)["heuristic-conservative"]
    for key in ("facts_hit", "efficiency", "breadth", "decisiveness", "cheating"):
        assert key in mt, f"缺少指标 {key}"
    assert 0.0 <= mt["efficiency"] <= 1.0
    assert 0.0 <= mt["decisiveness"] <= 1.0


def test_profile_excludes_cheating_by_default():
    mid = game.create_match("ts-003", ["mock", "heuristic-conservative"], max_rounds=20)
    game.run(mid)
    models = set(compute_profile()["model"])
    assert "mock" not in models, "作弊玩家不应出现在能力画像中"


# --- 排行榜作弊排除 ------------------------------------------------------
def test_leaderboard_excludes_cheating_by_default():
    mid = game.create_match("ts-004", ["mock"], max_rounds=10)
    game.run(mid)

    assert "mock" not in set(compute_leaderboard()["model"]), "默认排行榜不应含作弊玩家"
    assert "mock" in set(compute_leaderboard(exclude_cheating=False)["model"]), (
        "显式不排除时应能看到作弊玩家"
    )


# --- 雷达图导出 ----------------------------------------------------------
def test_radar_chart_is_generated(tmp_path: Path):
    mid = game.create_match(
        "ts-001", ["heuristic-conservative", "heuristic-aggressive"], max_rounds=24
    )
    game.run(mid)
    profile = compute_profile()
    out = generate_radar(profile, tmp_path / "radar.png")
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0, "雷达图文件为空"
