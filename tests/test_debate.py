"""辩论模式测试：比较制结算、平局加时赛、论证指标。

辩论（debate）与问答（qa）的本质区别是**比较制**：不看绝对阈值，
看相对对手谁更有说服力。这一组测试锁定三条行为：

1. 比较制——只要一方分数 > 0 即可决出胜负，不需要达到 qa 的绝对阈值；
2. 平局加时赛——分数并列时按论据充分度（debate）/ 提问效率（qa）决胜，
   副标准也持平则无人胜出，绝不按出手顺序送胜；
3. 收网动作——debate 模式下提交的是 argue（论证）而不是 guess（猜想）。
"""
import pytest

from mcp_server.engine import game
from mcp_server.engine.game import _settle_round
from mcp_server.eval.metrics import debate_metrics, match_metrics


# --- 比较制结算 ------------------------------------------------------------
def test_settle_round_debate_is_relative_not_absolute():
    """debate 模式下双方分数都低于 qa 阈值，但相对更高者仍然获胜。"""
    s = {"mode": "debate", "pending_guesses": {"a": 0.2, "b": 0.4}}
    assert _settle_round(s) == "b"


def test_settle_round_debate_zero_score_no_winner():
    """双方论证都没有命中任何线索（全 0），比较制也不该造出胜者。"""
    s = {"mode": "debate", "pending_guesses": {"a": 0.0, "b": 0.0}}
    assert _settle_round(s) is None


# --- 平局加时赛 ------------------------------------------------------------
def test_settle_round_debate_tie_broken_by_evidence():
    """debate 平分：论据（yes 提问数）更多的一方更有说服力。"""
    s = {
        "mode": "debate",
        "pending_guesses": {"a": 0.5, "b": 0.5},
        "transcript": [
            {"player": "a", "action": "ask", "host_answer": "yes"},
            {"player": "b", "action": "ask", "host_answer": "yes"},
            {"player": "b", "action": "ask", "host_answer": "yes"},
        ],
    }
    assert _settle_round(s) == "b"


def test_settle_round_debate_full_tie_yields_no_winner():
    """debate 分数与论据全部持平：无人胜出，不按出手顺序送胜。"""
    s = {
        "mode": "debate",
        "pending_guesses": {"a": 0.5, "b": 0.5},
        "transcript": [
            {"player": "a", "action": "ask", "host_answer": "yes"},
            {"player": "b", "action": "ask", "host_answer": "yes"},
        ],
    }
    assert _settle_round(s) is None


def test_settle_round_qa_tie_broken_by_efficiency():
    """qa 平分：同样命中率下提问更少的一方效率更高。"""
    s = {
        "pending_guesses": {"a": 0.7, "b": 0.7},
        "scores": {"a": {"questions": 6}, "b": {"questions": 3}},
    }
    assert _settle_round(s) == "b"


def test_settle_round_qa_full_tie_yields_no_winner():
    """qa 分数与提问数全部持平：无人胜出，比赛应继续。"""
    s = {
        "pending_guesses": {"a": 0.7, "b": 0.7},
        "scores": {"a": {"questions": 4}, "b": {"questions": 4}},
    }
    assert _settle_round(s) is None


# --- 端到端 -----------------------------------------------------------------
def test_debate_match_runs_end_to_end():
    """debate 模式可创建、可跑完，且收网动作是 argue。"""
    mid = game.create_match(
        "ts-002", ["heuristic-conservative", "heuristic-aggressive"],
        mode="debate", max_rounds=12,
    )
    st = game.run(mid)
    assert st["status"] == "finished"
    assert st["mode"] == "debate"
    argues = [r for r in st["transcript"] if r["action"] == "argue"]
    assert argues, "debate 模式下应有 argue 动作"
    assert all(r.get("score") is not None for r in argues), "论证应被打分"
    # 无论是否分出胜负，都不允许出现"分数相同却宣布胜者"的情况
    if st["winner"] is not None:
        scores = [sc["best_guess"] for m, sc in st["scores"].items() if m != st["winner"]]
        assert all(st["scores"][st["winner"]]["best_guess"] >= sc for sc in scores)


def test_debate_metrics_expose_persuasion():
    """辩论指标：论证命中率、说服力（相对值）、论据数、排名。"""
    mid = game.create_match(
        "ts-002", ["heuristic-conservative", "heuristic-aggressive"],
        mode="debate", max_rounds=12,
    )
    st = game.run(mid)
    dm = debate_metrics(st)
    assert dm["mode"] == "debate"
    for key in ("argument_accuracy", "persuasion", "evidence_used", "ranking"):
        assert key in dm, f"缺少辩论指标 {key}"
    # 说服力是相对值：最高者必须是 1.0，其余按比例缩放
    best = max(dm["argument_accuracy"].values())
    if best > 0:
        assert max(dm["persuasion"].values()) == pytest.approx(1.0)
    assert set(dm["ranking"]) == set(st["players"])


def test_debate_metrics_reject_non_debate_mode():
    """非 debate 模式不返回辩论指标（返回空 dict，不污染报告）。"""
    assert debate_metrics({"mode": "turtle_soup_qa"}) == {}


def test_analyze_match_attaches_debate_block():
    """analyze_match 在 debate 模式下应附带顶层 debate 指标块。"""
    from mcp_server import server as srv

    created = srv.start_match(
        "ts-002", ["heuristic-conservative", "heuristic-aggressive"],
        mode="debate", max_rounds=12,
    )
    mid = created["match_id"]
    srv.run_match(mid)
    an = srv.analyze_match(mid)
    assert "debate" in an, "debate 模式分析应包含 debate 指标块"
    assert an["debate"]["mode"] == "debate"
    assert set(an["debate"]["ranking"]) == set(an["debate"]["argument_accuracy"])


# --- 风格名解析回归 --------------------------------------------------------
def test_style_name_parsing_accepts_colon_and_hyphen():
    """heuristic:aggressive 与 heuristic-aggressive 必须解析到同一风格。

    曾经只认连字符：冒号写法静默 fallback 成 balanced，两个"不同性格"的
    玩家行为逐字相同，辩论全部平局。此测试锁定两种分隔符等价。
    """
    from mcp_server.engine.player import get_player
    from mcp_server.engine.puzzles import get_puzzle

    puzzle = get_puzzle("ts-001", include_truth=True)
    a = get_player("heuristic:aggressive", puzzle)
    b = get_player("heuristic-aggressive", puzzle)
    assert a.style_key == b.style_key == "aggressive"
    assert get_player("heuristic", puzzle).style_key == "balanced"
    assert get_player("heuristic:nonexistent", puzzle).style_key == "balanced"
