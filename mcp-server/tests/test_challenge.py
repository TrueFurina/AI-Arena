"""质疑（challenge）机制测试。

测谎原理：求解者重复提问同一个问题，若前后回答不一致即抓到谎言。
由此「伪装」第一次有了精确定义——**撒了谎但没被抓到**。

注意：misleading（稳定说反话）不会产生前后矛盾，因此重复提问法抓不到它；
这是求解者能力的真实局限，不是 bug。测试里对此做了明确断言。
"""
import random

from mcp_server.engine import game
from mcp_server.eval.metrics import deception_metrics
from mcp_server.store import logs


def _run(strategy: str, solver: str, seed: int, puzzle: str = "ts-001") -> dict:
    random.seed(seed)
    mid = game.create_match(
        puzzle, [f"insider-{strategy}", f"heuristic-{solver}"],
        max_rounds=20, mode="deception",
    )
    game.run(mid)
    return logs.load_match(mid)


def _avg_challenges(solver: str, strategy: str = "mixed", games: int = 6) -> float:
    total = 0
    for i in range(games):
        st = _run(strategy, solver, seed=100 + i)
        total += st.get("challenges", {}).get("total", 0)
    return total / games


def test_honest_insider_never_gets_challenged():
    """诚实者回答始终一致，重复提问也找不到矛盾，因此不该被质疑。"""
    st = _run("honest", "conservative", seed=11)
    assert st["challenges"]["total"] == 0, "诚实知情者被误质疑了"


def test_conservative_challenges_more_than_aggressive():
    """保守派复核率高（0.30），理应比几乎不复核的激进派（0.05）抓到更多谎言。"""
    cons = _avg_challenges("conservative")
    aggr = _avg_challenges("aggressive")
    assert cons > aggr, f"复核倾向没有转化为识破能力：保守 {cons} vs 激进 {aggr}"


def test_challenge_succeeds_whenever_conflict_is_found():
    """一旦发现前后不一致，核验就必须成立——否则质疑机制是假的。"""
    for seed in range(30):
        st = _run("mixed", "conservative", seed=seed)
        ch = st.get("challenges", {})
        if ch.get("total"):
            assert ch["success"] == ch["total"], (
                f"发现了矛盾却没判为谎言：{ch}"
            )
            return
    raise AssertionError("30 次尝试均未触发质疑，测谎机制可能失效")


def test_misleading_cannot_be_caught_by_repeat_questioning():
    """稳定说反话不产生前后矛盾，重复提问法对它无效（已知局限，非 bug）。"""
    for seed in (1, 2, 3):
        st = _run("misleading", "conservative", seed=seed)
        assert st["challenges"]["total"] == 0
        # 但它确实把求解者彻底骗倒了
        d = deception_metrics(st)
        assert d["avg_solver_accuracy"] == 0.0


def test_disguise_score_is_none_without_lies():
    """一次谎都没撒，谈不上伪装，应为 None 而非满分。"""
    st = _run("honest", "conservative", seed=5)
    d = deception_metrics(st)
    assert d["lies"] == 0
    assert d["disguise_score"] is None


def test_deception_metrics_exposes_disguise_fields():
    st = _run("mixed", "conservative", seed=7)
    d = deception_metrics(st)
    for key in ("caught", "caught_rate", "undetected_lies",
                "disguise_score", "challenge_accuracy"):
        assert key in d, f"缺少伪装指标 {key}"
    assert 0.0 <= d["caught_rate"] <= 1.0
