"""欺骗模式（deception）测试。

这是目前唯一能观测「欺骗 / 伪装」两个维度的入口，重点验证四件事：
1. 谎言确实被记录（否则算不出欺骗率）
2. 欺骗真的能压低求解者成绩（否则指标没意义）
3. 撒谎记录不泄漏给玩家（否则求解者等于开挂）
4. 知情者不参与求解者排名（他根本不是求解者）
"""
from mcp_server.engine import game
from mcp_server.eval.metrics import deception_metrics, is_insider, is_rankable
from mcp_server.store import logs


def _run_deception(strategy: str, puzzle_id: str = "ts-001") -> str:
    """跑一局欺骗模式：首位是知情者，后两位是求解者。"""
    mid = game.create_match(
        puzzle_id,
        [f"insider-{strategy}", "heuristic-conservative", "heuristic-aggressive"],
        max_rounds=20,
        mode="deception",
    )
    game.run(mid)
    return mid


def test_misleading_insider_lies_all_the_time():
    mid = _run_deception("misleading")
    log = logs.load_match(mid).get("_lie_log", [])
    assert log, "撒谎记录为空，无法计算欺骗率"
    assert all(r["lied"] for r in log)
    # 撒谎 = 给出的答案与真实答案相反
    assert all(r["given_answer"] != r["truth_answer"] for r in log)


def test_honest_insider_never_lies():
    mid = _run_deception("honest")
    log = logs.load_match(mid).get("_lie_log", [])
    assert log, "对照组也应有应答记录"
    assert not any(r["lied"] for r in log)


def test_lie_log_is_persisted_but_hidden_from_players():
    """撒谎记录要落盘供评测，但绝不能出现在玩家可见状态里。"""
    mid = _run_deception("mixed")
    assert "_lie_log" in logs.load_match(mid), "落盘应保留撒谎记录供离线评测"
    assert "_lie_log" not in game.get_state(mid), "玩家可见状态泄漏了撒谎记录"


def test_deception_actually_lowers_solver_accuracy():
    """欺骗必须真的能压低求解者命中率，否则这个维度就没有意义。"""
    def avg_acc(strategy: str) -> float:
        mid = _run_deception(strategy)
        return deception_metrics(logs.load_match(mid))["avg_solver_accuracy"]

    honest, misleading = avg_acc("honest"), avg_acc("misleading")
    assert honest > misleading, (
        f"欺骗未生效：诚实对照组 {honest} 应高于欺骗组 {misleading}"
    )


def test_deception_metrics_shape():
    mid = _run_deception("mixed")
    d = deception_metrics(logs.load_match(mid))
    for key in ("insider", "answers", "lies", "deception_rate", "avg_solver_accuracy"):
        assert key in d, f"缺少欺骗指标 {key}"
    assert 0.0 <= d["deception_rate"] <= 1.0


def test_insider_is_not_a_solver():
    """知情者是角色不是求解者，不应混入求解者排名。"""
    assert is_insider("insider-mixed") is True
    assert is_rankable("insider-mixed") is False
    assert is_rankable("heuristic-conservative") is True


def test_deception_requires_at_least_one_solver():
    try:
        game.create_match("ts-001", ["insider-mixed"], mode="deception")
    except ValueError as e:
        assert "至少需要" in str(e)
    else:
        raise AssertionError("只有知情者时应拒绝开局")
