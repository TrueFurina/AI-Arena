"""语义化动作空间（engine/semantic.py）的协议守护与实测局限。

分两类断言，性质不同，勿混淆：
  A. 协议正确性——无论策略多强都必须成立，违反即评测无效。
     其中「类别答案与 golden_qa 解耦」是最要紧的一条：它守住本设计
     不沦为循环论证（用我标注的答案去生成「客观」信号）。
  B. 已知局限——记录**当前**实测行为。真正实现强泛化后这些断言应失败，
     那是好事：届时带新数据更新本文件，而不是把断言删掉。
"""
import copy

import pytest

from engine.puzzles import load_puzzles
from engine.semantic import (
    CATEGORY_LEXICON, CATEGORY_NAMES, CategoryHost, category_answer,
    category_members, coverage_report,
)
from engine.evaluate import distinct_hit_rate
from players.rl import RLQuestioner, TabularQPolicy, run_episode
from eval_generalization import (
    DistinctRandomPolicy, FixedOrderPolicy, _prior_order,
    evaluate_policy, make_env, split_puzzles, train_policy,
)

MAX_ROUNDS = 8
TRAIN_SIZE = 6          # 沿用跨 5 种子验证过的划分规模
PER_PUZZLE_EPISODES = 800


# ---------------- A. 协议正确性 ----------------
def test_every_puzzle_has_at_least_one_hit_category():
    """每题至少有一个类别答「是」，否则该题上语义动作拿不到任何信号。

    这是关键词动作时代踩过的坑的防线：当年硬编码动作库未收录题003 任何
    关键词，三分之一训练局无论怎么问都只拿惩罚，系统性污染 Q 估计。
    实测每题命中类别数 2~7。
    """
    rep = coverage_report(load_puzzles())
    zero = [r["soup_id"] for r in rep if r["n_yes"] == 0]
    assert not zero, (
        f"以下题目在全部 {len(CATEGORY_NAMES)} 个语义类别上均无命中 → "
        f"该题上策略无法获得任何学习信号：{zero}"
    )


def test_category_answer_is_independent_of_golden_qa():
    """类别答案只由汤底文本派生，与 golden_qa 的 answer 字段无关（核心）。

    为什么这条最要紧：golden_qa 的 answer 是我手工标注的。若类别答案从它
    派生，就等于把人工判断包装成客观信号——用「我标注的答案」去证明
    「策略学到了东西」，属循环论证。这里把 golden_qa 全部翻转/清空后，
    类别答案必须逐位不变。
    """
    for p in load_puzzles():
        baseline = {c: category_answer(p.solution, c) for c in CATEGORY_NAMES}

        flipped = copy.replace(p, golden_qa=[
            {**qa, "answer": "无关" if qa["answer"] != "无关" else "是"}
            for qa in p.golden_qa
        ])
        cleared = copy.replace(p, golden_qa=[])
        for variant, name in ((flipped, "翻转 answer"), (cleared, "清空 golden_qa")):
            for c in CATEGORY_NAMES:
                assert category_answer(variant.solution, c) == baseline[c], (
                    f"{p.soup_id}：{name}后类别「{c}」的答案变了 → "
                    "类别答案依赖了 golden_qa，本设计退化为循环论证"
                )


def test_category_answer_depends_on_solution():
    """上一条的反面：改汤底必须能改变类别答案（证明它真的读了汤底）。

    只断言「与 golden_qa 无关」是不够的——若实现写死返回「是」，上一条
    也会通过。这里要求至少存在一题一类别，改汤底后答案翻转。
    """
    changed = 0
    for p in load_puzzles():
        emptied = copy.replace(p, solution="")
        for c in CATEGORY_NAMES:
            if category_answer(p.solution, c) != category_answer(emptied.solution, c):
                changed += 1
                break
    assert changed > 0, (
        "清空汤底后没有任何类别答案改变 → category_answer 没在读 solution，"
        "上一条「与 golden_qa 无关」的断言很可能只是因为函数写死了"
    )


def test_unknown_category_is_not_silently_confirmed():
    """未知类别：词表查询直接报错，主持人答「无关」——绝不静默当「是」。

    静默 fallback 比报错危险（项目已有教训）：若未知类别被当成「是」，
    策略会学到虚假的正反馈；若被当成空词表「否」，同样掩盖配置错误。
    """
    with pytest.raises(KeyError):
        category_members("不存在的类别")
    host = CategoryHost()
    p = load_puzzles()[0]
    assert host.answer(p, "这件事和「不存在的类别」有关吗？", []) == "无关"
    assert host.answer(p, "没有类别标记的闲聊", []) == "无关"


def test_semantic_bank_is_puzzle_independent():
    """语义动作库必须与题目无关（否则又变成题目绑定的关键词动作）。

    这是「迁移路径存在」的前提：动作库若在训练题与新题上不同，新题上
    的动作就是没训练过的动作，退化回关键词模式的老问题。
    """
    puzzles = load_puzzles()
    full, _ = make_env("semantic", puzzles)[0], None
    assert len(full) == len(CATEGORY_NAMES)
    for n in (3, 6, 9):
        train, held = split_puzzles(puzzles, n, seed=0)
        b_train, *_ = make_env("semantic", train)
        b_held, *_ = make_env("semantic", held)
        assert b_train == b_held == full, (
            f"n={n}：语义动作库随题目划分变化 → 动作与题目绑定，无迁移路径"
        )


def test_dhit_penalizes_repetition():
    """dhit 必须惩罚重复提问——这是它存在的理由（语义类别跨题共享）。

    语义类别在任意题上都能答「是」，重复问同一个命中类别会把 hit_rate
    刷到 1.0。dhit 的分母取**提问总轮数**而非「不同问句数」，故重复问
    只按 1 条线索计分。
    """
    def tr(questions, answers):
        out = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            out.append({"role": "question", "content": q})
            out.append({"role": "answer", "content": a})
        return out

    repeated = tr(["Q1"] * 8, ["是"] * 8)
    distinct = tr([f"Q{i}" for i in range(8)], ["是"] * 8)
    assert distinct_hit_rate(distinct) == 1.0
    assert distinct_hit_rate(repeated) == pytest.approx(1 / 8), (
        f"重复提问 8 次仍得 {distinct_hit_rate(repeated):.3f} → "
        "分母取了「不同问句数」，重复刷分没堵住"
    )
    half = tr([f"Q{i}" for i in range(8)], ["是"] * 4 + ["否"] * 4)
    assert distinct_hit_rate(half) == pytest.approx(0.5)


# ---------------- B. 已知局限（实测行为，改进后应失效） ----------------
def _train_and_score(mode, train, held, seed=0):
    """训练并测新题 dhit。返回 (新题 dhit, 训练集 dhit)。"""
    env = make_env(mode, load_puzzles())
    pol = train_policy(train, env, PER_PUZZLE_EPISODES * len(train), seed,
                       MAX_ROUNDS)
    _, _, held_d, _ = evaluate_policy(held, env, pol, repeats=5,
                                      max_rounds=MAX_ROUNDS)
    _, _, train_d, _ = evaluate_policy(train, env, pol, repeats=5,
                                       max_rounds=MAX_ROUNDS)
    return held_d, train_d, env, pol


def test_semantic_beats_keyword_on_unseen_puzzles():
    """语义动作在新题上显著优于关键词动作——迁移路径确实打开了。

    实测（n_train=6，跨 5 个种子）：
      语义  新题 dhit 0.417 ~ 0.562
      关键词（题目条件）新题 dhit 0.000 ~ 0.167
      差值 +0.271 ~ +0.479，5 个种子全部为正
    关键词动作在新题上几乎为零：它的动作是题目特有词，新题关键词不在
    训练过的状态里。语义动作跨题通用，故能拿到分。
    """
    puzzles = load_puzzles()
    train, held = split_puzzles(puzzles, TRAIN_SIZE, seed=0)
    sem_held, _sem_train, _, _ = _train_and_score("semantic", train, held)
    key_held, _key_train, _, _ = _train_and_score("puzzle", train, held)
    assert sem_held - key_held > 0.15, (
        f"语义 {sem_held:.3f} 相对关键词 {key_held:.3f} 的优势仅 "
        f"{sem_held - key_held:+.3f} → 语义化改造未带来预期收益，请复查"
    )


def test_semantic_gain_is_only_the_global_prior():
    """训练出的策略 ≈ 「按全局频率问」的参考值 → 学到的只有全局先验。

    实测（n_train=6，跨 5 个种子）：|训练策略 − 先验顺序| ≤ 0.104。
    含义：语义化动作打开了迁移路径，但迁移过来的信息量很小——只有
    「哪些类别通常更有用」这一条全局统计，没有针对具体题目的判断。
    真正的下一步是让状态带上**汤面的语义特征**（而非题目 ID），
    使策略能按题调整，又不必记住题目。

    ⚠️ 若后续实现了按题自适应的泛化，本断言应失败（训练值会明显超过
    先验上限），届时请带新数据更新，而不是放宽阈值。
    """
    puzzles = load_puzzles()
    train, held = split_puzzles(puzzles, TRAIN_SIZE, seed=0)
    sem_held, _train_d, env, _pol = _train_and_score("semantic", train, held)
    bank = env[0]
    prior = FixedOrderPolicy(_prior_order(train, bank))
    _, _, prior_held, _ = evaluate_policy(held, env, prior, repeats=5,
                                          max_rounds=MAX_ROUNDS)
    assert abs(sem_held - prior_held) < 0.20, (
        f"训练策略新题 dhit {sem_held:.3f} 与先验顺序 {prior_held:.3f} 相差 "
        f"{abs(sem_held - prior_held):.3f} → 超出「只学到全局先验」的范围，"
        "请更新本局限描述（这可能正是期待中的进展）"
    )
    # 顺带守住：先验确实优于随机不重复，否则这条局限描述本身就无意义
    rnd = DistinctRandomPolicy(len(bank), seed=0)
    _, _, rnd_held, _ = evaluate_policy(held, env, rnd, repeats=20,
                                        max_rounds=MAX_ROUNDS)
    assert prior_held > rnd_held, (
        f"先验顺序 {prior_held:.3f} 未优于随机不重复 {rnd_held:.3f} → "
        "全局频率里没有可用信息，本局限描述的前提不成立"
    )


def test_semantic_does_not_memorize_train_puzzles():
    """语义模式下训练题与新题表现接近——不像题目条件模式那样靠记忆。

    对比（n_train=6，seed=0）：
      题目条件：训练 0.458 / 新题 0.021，落差 +0.438 → 典型记忆
      语义    ：训练 0.375 / 新题 0.562，落差 −0.188 → 落差小甚至为负
    落差小是「没有过拟合」的证据，但**不等于泛化强**（见上一条：
    它只是在新题上表现得像在训练题上一样，水平本身受限于全局先验）。
    """
    puzzles = load_puzzles()
    train, held = split_puzzles(puzzles, TRAIN_SIZE, seed=0)
    sem_held, sem_train, _, _ = _train_and_score("semantic", train, held)
    assert abs(sem_train - sem_held) < 0.25, (
        f"语义模式训练题 {sem_train:.3f} 与新题 {sem_held:.3f} 落差 "
        f"{abs(sem_train - sem_held):.3f} → 出现记忆式过拟合，请复查状态表示"
    )
