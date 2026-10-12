"""L2 执行层测试（players/rl.py）。

重点不是覆盖率，而是回答一个硬问题：**Q 表到底学到东西了吗？**
训练曲线上升本身证明不了学习（可能只是探索噪声）。因此这里用
「同题库、同奖励函数、固定种子」下的贪心 vs 随机对比作为学习证据。
"""
import pytest

from engine.game import TurtleSoupGame
from engine.puzzles import load_puzzles
from players.base import Player
from players.mock import MockHost
from players.rl import (
    ACTION_BANK, RLQuestioner, TabularQPolicy, render_question, run_episode,
)


def test_rlquestioner_implements_player():
    q = RLQuestioner()
    assert isinstance(q, Player)
    assert q.bank == ACTION_BANK


def test_render_question_contains_keyword():
    """渲染出的问句必须含关键词：MockHost 按子串匹配 golden_qa，
    关键词不在问句里就拿不到有效回答，RL 也就不存在学习信号。"""
    for kw in ACTION_BANK:
        q = render_question(kw)
        assert kw in q
        assert q.endswith("？")


def test_untrained_greedy_is_deterministic():
    """未训练时 Q 全 0，贪心按 tie-break 固定取 index 0（可复现性保障）。"""
    puzzles = load_puzzles()
    q = RLQuestioner(policy=TabularQPolicy(len(ACTION_BANK), seed=0), explore=False)
    out = run_episode(puzzles[0], MockHost(), q, max_rounds=3, train=False)
    questions = [m["content"] for m in out["transcript"] if m["role"] == "question"]
    assert len(questions) == 3
    assert all(ACTION_BANK[0] in x for x in questions)


def test_runs_inside_real_engine():
    """RL 提问者必须能直接进 TurtleSoupGame（与 LLM 玩家同一接口）。"""
    puzzles = load_puzzles()
    q = RLQuestioner(explore=False)
    game = TurtleSoupGame(puzzles[0], MockHost(), [q], judge=None, max_rounds=4)
    res = game.run()
    roles = [m["role"] for m in res.transcript]
    assert roles.count("question") == 4
    assert roles.count("answer") == 4
    assert roles.count("guess") == 1


def _mean_reward(policy, puzzles, repeats=10, explore=False, max_rounds=4, seed=0):
    host = MockHost()
    q = RLQuestioner(policy=policy, explore=explore)
    vals = []
    for _ in range(repeats):
        for p in puzzles:
            vals.append(run_episode(p, host, q, max_rounds=max_rounds,
                                    train=False)["reward"])
    return sum(vals) / len(vals)


def test_trained_policy_beats_random():
    """核心学习证据：训练后贪心策略奖励显著高于随机基线（固定种子）。

    可学习的两个信号：① 不重复已问动作；② 偏好信息型动作（避开 钱/水）。
    """
    puzzles = load_puzzles()
    policy = TabularQPolicy(len(ACTION_BANK), seed=0)
    q = RLQuestioner(policy=policy, explore=True)
    host = MockHost()

    for ep in range(200):
        p = puzzles[ep % len(puzzles)]
        run_episode(p, host, q, max_rounds=4, train=True)

    greedy = _mean_reward(policy, puzzles, explore=False, seed=0)
    random_p = TabularQPolicy(len(ACTION_BANK), epsilon=1.0, seed=0)
    rand = _mean_reward(random_p, puzzles, explore=True, seed=0)

    assert greedy > rand, (
        f"贪心({greedy:+.3f}) 未超过随机({rand:+.3f}) → 策略未学到有效信息"
    )
    # 学到的具体内容：**不重复**已问动作（经 5 个种子验证的稳定信号）。
    # 不要断言「偏好信息型动作」——该信号在本最小环境下学不出来（已证伪，
    # 详见 players/rl.py 模块文档）；断言它只会得到一个随机失败的假测试。
    row = policy.Q.get("0", [0.0] * len(ACTION_BANK))
    best_unasked = max(q for i, q in enumerate(row) if i != 0)
    assert row[0] < best_unasked, (
        "策略未学会不重复：已问动作的 Q 未低于未问动作 "
        f"(Q(state={{0}}, 动作0)={row[0]:.3f} >= 最优未问={best_unasked:.3f})"
    )


def test_policy_save_load_roundtrip(tmp_path):
    p = TabularQPolicy(len(ACTION_BANK), seed=0)
    p.update((), 1, 1.0, (1,))
    path = tmp_path / "pol.json"
    p.save(str(path))
    loaded = TabularQPolicy.load(str(path), seed=0)
    assert loaded.Q == p.Q
    assert loaded.n_actions == p.n_actions
