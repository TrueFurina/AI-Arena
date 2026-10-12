"""L2 执行层测试（players/rl.py）。

重点不是覆盖率，而是回答一个硬问题：**Q 表到底学到东西了吗？**
训练曲线上升本身证明不了学习（可能只是探索噪声）。因此这里用
「同题库、同奖励函数、固定种子」下的贪心 vs 随机对比作为学习证据，
并且同时看**代理奖励**与**客观 F1** 两个指标。

断言取舍（诚实纪律）：
  只断言经跨种子验证的稳健信号。已被证伪的预期（旧表示下「偏好有信息
  动作」）不再断言——断言它只会得到一个随机失败的假测试。修正状态表示
  后该信号已稳定可学，故重新纳入，但要求**跨 3 个种子全部成立**。
"""
import pytest

from engine.game import TurtleSoupGame
from engine.puzzles import load_puzzles
from engine.evaluate import char_f1
from players.base import Player
from players.mock import MockHost
from players.rl import (
    build_action_bank, RLQuestioner, TabularQPolicy,
    render_question, parse_question_keyword, run_episode,
)

MAX_ROUNDS = 8


def _bank():
    return build_action_bank(load_puzzles())


def _train(seed=0, episodes=200, bank=None, info=None):
    puzzles = load_puzzles()
    bank, info = bank or _bank()[0], info or _bank()[1]
    host = MockHost()
    policy = TabularQPolicy(len(bank), seed=seed)
    q = RLQuestioner(policy=policy, action_bank=bank, informative=info,
                     explore=True)
    for ep in range(episodes):
        run_episode(puzzles[ep % len(puzzles)], host, q,
                    max_rounds=MAX_ROUNDS, train=True)
    return policy, bank, info, puzzles


def _eval(policy, bank, info, puzzles, explore=False, seed=0, repeats=10):
    host = MockHost()
    q = RLQuestioner(policy=policy, action_bank=bank, informative=info,
                     explore=explore)
    rewards, f1s = [], []
    for _ in range(repeats):
        for p in puzzles:
            out = run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False)
            rewards.append(out["reward"])
            f1s.append(out["f1"])
    n = len(rewards)
    return sum(rewards) / n, sum(f1s) / n


def test_rlquestioner_implements_player():
    bank, info = _bank()
    q = RLQuestioner(action_bank=bank, informative=info)
    assert isinstance(q, Player)
    assert q.bank == bank


def test_action_bank_covers_every_puzzle():
    """防漏洞回归：硬编码动作库曾导致题003 上零有效动作。

    该题上无论怎么问都只拿到惩罚，系统性污染所有动作的 Q 估计。
    派生式动作库必须保证每题至少有一个「有信息」的动作可用。
    """
    puzzles = load_puzzles()
    bank, info = build_action_bank(puzzles)
    for p in puzzles:
        useful = [
            kw for kw in bank
            if any(qa["contains"] == kw and qa["answer"] != "无关"
                   for qa in p.golden_qa)
        ]
        assert useful, f"{p.soup_id} 上没有任何有效动作（动作库覆盖漏洞）"


def test_state_includes_puzzle_identity():
    """状态必须含题目标识：否则构成 POMDP，Q 只能收敛到跨题混合平均。

    这是「偏好有信息动作」信号早期学不出来的根因（表示不允许，非样本量）。
    """
    bank, info = _bank()
    puzzles = load_puzzles()
    q = RLQuestioner(action_bank=bank, informative=info, explore=False)
    keys = {q.state_key(p, []) for p in puzzles}
    assert len(keys) == len(puzzles), "不同题目的初始状态键相同 → 仍是 POMDP"


def test_render_question_contains_keyword():
    """渲染出的问句必须含关键词：MockHost 按子串匹配 golden_qa。"""
    bank, _ = _bank()
    for kw in bank:
        q = render_question(kw)
        assert kw in q and q.endswith("？")
        assert parse_question_keyword(q) == kw


def test_untrained_greedy_is_deterministic():
    bank, info = _bank()
    puzzles = load_puzzles()
    q = RLQuestioner(policy=TabularQPolicy(len(bank), seed=0),
                     action_bank=bank, informative=info, explore=False)
    out = run_episode(puzzles[0], MockHost(), q, max_rounds=3, train=False)
    questions = [m["content"] for m in out["transcript"] if m["role"] == "question"]
    assert len(questions) == 3
    assert all(bank[0] in x for x in questions)


def test_runs_inside_real_engine():
    """RL 提问者必须能直接进 TurtleSoupGame（与 LLM 玩家同一接口）。"""
    bank, info = _bank()
    puzzles = load_puzzles()
    q = RLQuestioner(action_bank=bank, informative=info, explore=False)
    game = TurtleSoupGame(puzzles[0], MockHost(), [q], judge=None, max_rounds=4)
    res = game.run()
    roles = [m["role"] for m in res.transcript]
    assert roles.count("question") == 4
    assert roles.count("answer") == 4
    assert roles.count("guess") == 1


def test_final_guess_is_evidence_driven():
    """F1 通路：还原质量必须对提问质量敏感。

    早期版本把「问句原文 + 答案 + 模板废话」整段拼进还原句，模板字几乎
    不出现在汤底里，把精确率压死，导致 F1 对提问质量不敏感——策略学到的
    东西体现不到还原质量上（贪心/随机 F1 差仅 0.02）。现改为只采纳被
    主持确认的线索，本测试守护这条通路不被改回去。
    """
    bank, info = _bank()
    puzzles = load_puzzles()
    q = RLQuestioner(policy=TabularQPolicy(len(bank)), action_bank=bank,
                     informative=info, explore=False)

    def transcript_for(keywords, answer):
        tr = []
        for kw in keywords:
            tr.append({"role": "question", "speaker": q.id,
                       "content": render_question(kw)})
            tr.append({"role": "answer", "speaker": "host", "content": answer})
        return tr

    for p in puzzles:
        yes = [qa["contains"] for qa in p.golden_qa if qa["answer"] == "是"]
        no = [qa["contains"] for qa in p.golden_qa if qa["answer"] == "无关"]
        g_yes = q.final_guess(p, transcript_for(yes[:MAX_ROUNDS], "是"))
        g_no = q.final_guess(p, transcript_for((no * MAX_ROUNDS)[:MAX_ROUNDS], "无关"))
        assert char_f1(g_yes, p.solution) > char_f1(g_no, p.solution), (
            f"{p.soup_id}：问中线索的还原 F1({char_f1(g_yes, p.solution):.3f}) "
            f"未高于全问无关({char_f1(g_no, p.solution):.3f}) → F1 通路断了"
        )
        # 精确率判据：只比 F1 方向不够严。变异验证时发现，把还原句改回
        # 「问句+答案+模板废话」的旧写法，F1 方向的断言**仍然通过**（模板
        # 字对两种输入都加，只是把精确率从 0.50~0.65 稀释到 0.25~0.32），
        # 回退防不住。故此处直接守精确率：还原句里不得掺杂汤底没有的废话。
        sol, gs = set(p.solution), set(g_yes)
        precision = len(gs & sol) / len(gs)
        assert precision > 0.45, (
            f"{p.soup_id}：还原句精确率 {precision:.3f} 过低 → "
            "还原句混入了汤底中不存在的模板字（如『RL 策略还原』『→』）"
        )


def test_final_guess_excludes_denied_clues():
    """被主持否定/判为无关的线索不得进入还原句（只采纳确认线索）。"""
    bank, info = _bank()
    p = load_puzzles()[0]
    q = RLQuestioner(policy=TabularQPolicy(len(bank)), action_bank=bank,
                     informative=info, explore=False)
    tr = [
        {"role": "question", "speaker": q.id, "content": render_question("肉")},
        {"role": "answer", "speaker": "host", "content": "是"},
        {"role": "question", "speaker": q.id, "content": render_question("海难")},
        {"role": "answer", "speaker": "host", "content": "无关"},
    ]
    guess = q.final_guess(p, tr)
    assert "肉" in guess and "海难" not in guess


def test_trained_policy_beats_random_on_reward_and_f1():
    """核心学习证据：贪心策略在**奖励**与**客观 F1** 上均超过随机基线。

    只比奖励不够——奖励是代理指标，可能被「不重复」这类取巧行为刷高。
    F1 是还原质量本身，两者同时成立才算真的学到了有用的提问策略。
    """
    policy, bank, info, puzzles = _train(seed=0, episodes=200)
    g_r, g_f1 = _eval(policy, bank, info, puzzles, explore=False)
    r_r, r_f1 = _eval(TabularQPolicy(len(bank), epsilon=1.0, seed=0),
                      bank, info, puzzles, explore=True)

    assert g_r > r_r, f"贪心奖励({g_r:+.3f}) 未超过随机({r_r:+.3f})"
    assert g_f1 > r_f1, (
        f"贪心 F1({g_f1:.3f}) 未超过随机({r_f1:.3f}) → "
        "策略学到的东西没有转化为还原质量提升"
    )


def test_informative_preference_is_stable_across_seeds():
    """偏好有信息动作：修正状态表示后已可学，要求跨 3 个种子全部成立。

    旧表示（状态不含题目）下该信号不可学——12 组配置中 8 组失败、排序随
    种子乱跳。根因是 POMDP，不是样本量。此测试同时也是该修复的回归守护。
    """
    for seed in (0, 1, 2):
        policy, bank, info, puzzles = _train(seed=seed, episodes=200)
        gaps = []
        for p in puzzles:
            row = policy.Q.get(f"{p.soup_id}|")
            if row is None:
                continue
            pos = [row[i] for i in info]
            neg = [row[i] for i in range(len(bank)) if i not in info]
            gaps.append(sum(pos) / len(pos) - sum(neg) / len(neg))
        assert gaps, f"seed={seed}：未产生题目条件下的 Q 状态"
        gap = sum(gaps) / len(gaps)
        assert gap > 0, (
            f"seed={seed}：有信息型-无信息型 平均 Q 差={gap:+.3f} ≤ 0 → "
            "策略未区分有信息与无信息动作"
        )


def test_avoids_repeating_asked_actions():
    """不重复已问动作：经多种子验证的稳定信号。"""
    policy, bank, info, puzzles = _train(seed=0, episodes=200)
    p = puzzles[0]
    row = policy.Q.get(f"{p.soup_id}|0", [0.0] * len(bank))
    best_unasked = max(row[i] for i in range(len(bank)) if i != 0)
    assert row[0] < best_unasked, (
        f"策略未学会不重复：Q(已问动作0)={row[0]:.3f} >= 最优未问={best_unasked:.3f}"
    )


def test_policy_save_load_roundtrip(tmp_path):
    bank, _ = _bank()
    p = TabularQPolicy(len(bank), seed=0)
    p.update("x|", 1, 1.0, "x|1")
    path = tmp_path / "pol.json"
    p.save(str(path))
    loaded = TabularQPolicy.load(str(path), seed=0)
    assert loaded.Q == p.Q
    assert loaded.n_actions == p.n_actions
