"""泛化评测协议 + 题库质量约束（守护 eval_generalization.py 的前提）。

分两类断言，性质不同，勿混淆：
  A. 协议正确性——无论策略多强都必须成立，违反即评测无效
  B. 已知局限——记录**当前**设计下的实测行为。若后续做了语义化动作
     改造真正提升了泛化，这些断言会失败，那是**预期的、好事**：
     届时请带着新数据更新本文件，而不是把断言删掉。
"""
import pytest

from engine.puzzles import load_puzzles
from engine.evaluate import hit_rate
from players.mock import MockHost
from players.rl import (
    build_action_bank, RLQuestioner, TabularQPolicy, run_episode,
)
from eval_generalization import split_puzzles, evaluate_policy

MAX_ROUNDS = 8


def _full_bank():
    puzzles = load_puzzles()
    return puzzles, *build_action_bank(puzzles)


# ---------------- A. 协议正确性 ----------------
def test_split_is_disjoint_and_complete():
    """held-out 划分必须无泄漏：训练题与新题不相交，且并集为全集。"""
    puzzles = load_puzzles()
    for n in (3, 6, 9):
        train, held = split_puzzles(puzzles, n, seed=0)
        assert len(train) == n
        assert len(held) == len(puzzles) - n
        assert not (set(id(p) for p in train) & set(id(p) for p in held)), (
            f"n={n}：训练题与新题有交集 → held-out 泄漏，评测无效"
        )
        assert len(train) + len(held) == len(puzzles)


def test_action_bank_is_vocabulary_not_leak():
    """动作库按「词汇表」处理：从全题库派生，不含答案信息。

    若改成只从训练题派生，测的就变成动作空间覆盖率而非策略泛化，
    答非所问（新题关键词不在库里 → 必然 0 分，与策略能力无关）。
    """
    puzzles = load_puzzles()
    bank_full, _ = bank_info = build_action_bank(puzzles)
    train, _held = split_puzzles(puzzles, 6, seed=0)
    bank_train, _ = build_action_bank(train)
    assert len(bank_full) > len(bank_train), (
        "动作库未覆盖全题库 → 退化为只测覆盖率"
    )


# ---------------- B. 已知局限（实测行为，改进后应失效） ----------------
def test_puzzle_mode_has_zero_generalization_gain():
    """题目条件模式：训练题上学会，新题上**与未训练策略逐位相同**。

    实测（12 题库 / 92 动作 / 每题 800 局，训练题数 3、6、9）：
      训练题命中率 0.500 / 0.604 / 0.417
      新题命中率   = 未训练基线（0.111 / 0.167 / 0.333）
      真泛化增益   = +0.000 / +0.000 / +0.000
    原因是数学必然的：新题的状态键（soup_id|...）在训练中从未出现，
    Q 全零 → 行为退化为未训练策略，增益精确为 0，不是「接近 0」。

    ⚠️ 语义化动作改造一旦真正带来泛化，本断言应失败——届时请更新。
    """
    puzzles, bank, info = _full_bank()
    train, held = split_puzzles(puzzles, 6, seed=0)
    host = MockHost()

    policy = TabularQPolicy(len(bank), seed=0)
    q = RLQuestioner(policy=policy, action_bank=bank, informative=info,
                     explore=True, use_puzzle_state=True)
    # 800 局/题：与 eval_generalization.py 默认一致。400 局/题时训练题命中率
    # 仅 0.188（欠拟合），会让「训练题 vs 新题落差」断言因训练不足而误判。
    for ep in range(800 * len(train)):
        run_episode(train[ep % len(train)], host, q, max_rounds=MAX_ROUNDS,
                    train=True)

    _, trained_hit, _ = evaluate_policy(held, bank, info, policy, "puzzle",
                                        repeats=5, max_rounds=MAX_ROUNDS)
    baseline = TabularQPolicy(len(bank), seed=0)
    _, base_hit, _ = evaluate_policy(held, bank, info, baseline, "puzzle",
                                     repeats=5, max_rounds=MAX_ROUNDS)
    assert abs(trained_hit - base_hit) < 1e-9, (
        f"新题命中率 {trained_hit:.4f} 与未训练基线 {base_hit:.4f} 不同 → "
        "题目条件模式出现了非预期的泛化（若属实，请更新本局限描述）"
    )
    # 第二条判据（有辨别力）：上面那条近乎「结构性恒真」——只要新题状态键
    # 未被训练过就必然成立，无法反映训练是否真的有效。故补充落差断言：
    # 训练题与新题必须存在显著落差。若将来真实现了泛化，落差会收窄，
    # 本断言会失败并提醒更新，而不是让一条恒真断言一直假装通过。
    _, train_hit, _ = evaluate_policy(train, bank, info, policy, "puzzle",
                                      repeats=5, max_rounds=MAX_ROUNDS)
    assert train_hit - trained_hit > 0.15, (
        f"训练题命中率 {train_hit:.3f} 与新题 {trained_hit:.3f} 落差仅 "
        f"{train_hit - trained_hit:+.3f} → 要么训练无效，要么已具备泛化，"
        "二者都需更新本局限描述"
    )


def test_global_mode_train_performance_is_weak():
    """题目无关模式：连训练题都学不好——动作价值题目绑定时跨题平均无差异。

    实测训练题命中率随训练题数增加反而下降（0.125 → 0.042 → 0.014），
    这与理论一致：训练题越多，跨题平均越均匀，Q 差异越小 → 趋近随机。
    """
    puzzles, bank, info = _full_bank()
    train, _held = split_puzzles(puzzles, 6, seed=0)
    host = MockHost()

    policy = TabularQPolicy(len(bank), seed=0)
    q = RLQuestioner(policy=policy, action_bank=bank, informative=info,
                     explore=True, use_puzzle_state=False)
    for ep in range(400 * len(train)):
        run_episode(train[ep % len(train)], host, q, max_rounds=MAX_ROUNDS,
                    train=True)
    _, hit, _ = evaluate_policy(train, bank, info, policy, "global", repeats=5,
                                max_rounds=MAX_ROUNDS)
    # 只守上限：题目无关模式在当前动作抽象下不可能很强
    assert hit < 0.40, (
        f"题目无关模式训练题命中率 {hit:.3f} 偏高 → 若属实说明已具备泛化，请更新"
    )


# ---------------- 题库质量约束（防回归） ----------------
def test_no_substring_conflict_within_puzzle():
    """同一题内关键词不得互为子串。

    MockHost 按 golden_qa 顺序做子串匹配并取第一个命中。若「毒药」与
    「下毒」共存这类情况尚可，但「肉」与「肉汤」共存会让后者永远匹配
    到前者的答案，属静默错误（不报错、直接给错答案）。
    """
    for p in load_puzzles():
        kws = [qa["contains"] for qa in p.golden_qa]
        for i, a in enumerate(kws):
            for j, b in enumerate(kws):
                if i != j and a in b:
                    pytest.fail(
                        f"{p.soup_id}：「{a}」是「{b}」的子串 → 主持匹配会静默误判"
                    )


def test_every_puzzle_has_confirmable_clues():
    """每题至少有一条答案为「是」的线索，否则该题永远无法被问中。"""
    for p in load_puzzles():
        assert any(qa["answer"] == "是" for qa in p.golden_qa), (
            f"{p.soup_id}：没有可确认线索 → 策略在该题上必然 0 分"
        )


def test_confirm_keywords_appear_in_solution():
    """答案为「是」的关键词应尽量出现在汤底里。

    否则「问中了但汤底用词不同」会稀释 F1，让还原质量指标无法反映提问
    质量——这也是要把 hit_rate 与 f1 分开报的原因。此处守的是题库质量：
    允许极少数例外，但不允许大面积措辞脱节。
    """
    offenders = []
    for p in load_puzzles():
        yes = [qa["contains"] for qa in p.golden_qa if qa["answer"] == "是"]
        missing = [k for k in yes if k not in p.solution]
        if len(missing) > 2:
            offenders.append(f"{p.soup_id}: {missing}")
    assert not offenders, (
        "以下题目的可确认关键词大面积未出现在汤底中（措辞脱节，会压死 F1）："
        + "; ".join(offenders)
    )


def test_hit_rate_metric_semantics():
    """hit_rate 只统计被确认的回答（与 yn_rate 不同，后者含「否」）。"""
    assert hit_rate(["是", "是"]) == 1.0
    assert hit_rate(["是", "否"]) == 0.5
    assert hit_rate(["无关", "无关"]) == 0.0
    assert hit_rate([]) == 0.0
