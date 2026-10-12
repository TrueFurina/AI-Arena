"""P2-B MARL 主持侧守护（players/host_rl.py + run_episode 主持钩子 + tdhit）。

与项目纪律一致：**全绿 ≠ 有效**。关键断言配变异验证（提交信息记录本轮
六条变异及其击杀测试），重点守护：

  ① 配额强制——政策永远选 lie 也只恰好骗 k 次（否则分配问题退化）
  ② 状态题目无关——同真相不同题 → 同状态键（迁移能力来自构造）
  ③ 奖励零和镜像——主持总奖励 = −提问者总奖励 − λ·谎言数（精确等式）
  ④ tdhit 不被谎言刷高——dhit 对「否→是」的谎言照单全收，tdhit 不计
  ⑤ 冲突丢弃恒等性——诚实主持下还原句与旧行为逐位相同（跨种子）
  ⑥ run_episode 默认路径——键集合精确为旧四个，数值可独立重算复现
  ⑦ tdhit 恒等式——每谎伤害 ≡ 1/max_rounds（§6-A 发现 1：原 E1 判据
     因此弃用；本测试守护该退化依据，tdhit 语义被改坏时立即变红）
"""
import pytest

from engine.game import Puzzle, TurtleSoupGame
from engine.puzzles import load_puzzles
from engine.evaluate import char_f1, distinct_hit_rate, tdhit
from engine.semantic import (
    CATEGORY_NAMES, CategoryHost, category_answer, parse_category,
)
from players.host_rl import (
    ACTION_LIE, DeceptiveHost, FirstKLiarHost, RandomLiarHost,
    category_truth_fn, invert_answer,
)
from players.mock import MockHost, MockJudge, MockQuestioner
from players.rl import (
    LIE_COST_DEFAULT, RLQuestioner, TabularQPolicy, render_question,
    parse_question_keyword, run_episode, step_reward, terminal_reward,
)

MAX_ROUNDS = 8
BANK = list(CATEGORY_NAMES)
N_CAT = len(BANK)


def _puzzle(idx: int = 0):
    return load_puzzles()[idx]


def _questioner(policy=None, explore=False):
    return RLQuestioner(policy=policy or TabularQPolicy(N_CAT, seed=0),
                        action_bank=BANK, explore=explore,
                        use_puzzle_state=False)


def _truth_fn(puzzle):
    def tf(question):
        cat = parse_category(question)
        return None if cat is None else category_answer(puzzle.solution, cat)
    return tf


class _AlwaysLiePolicy:
    """桩策略：永远选 lie——用于配额强制/镜像测试（免去先训练出爱说谎的 Q）。"""

    n_actions = 2
    gamma = 0.9

    def __init__(self):
        self.updates = []

    def select_action(self, state, explore=True):
        return ACTION_LIE

    def greedy_action(self, state):
        return ACTION_LIE

    def update(self, state, action, reward, next_state, done=False):
        self.updates.append((state, action, reward))


class _NoRepeatPolicy:
    """桩策略：总是问第一个未问过的动作（免训练的「不重复」提问者）。

    用于镜像符号测试：重复提问会拿 −0.5 惩罚，把提问者总奖励压成负数，
    镜像后主持反而为正——符号断言必须建立在「提问者表现正常」的对局上。
    """

    n_actions = N_CAT
    gamma = 0.9

    def select_action(self, state, explore=True):
        asked = {int(x) for x in str(state).split(",")
                 if x.strip().lstrip("-").isdigit()}
        for i in range(self.n_actions):
            if i not in asked:
                return i
        return 0

    def greedy_action(self, state):
        return self.select_action(state)

    def update(self, *a, **k):
        pass


def _pairs(transcript):
    """[(question, answer)] 按序展开（跳过 guess）。"""
    out, pending = [], None
    for m in transcript:
        if m.get("role") == "question":
            pending = m.get("content", "")
        elif m.get("role") == "answer" and pending is not None:
            out.append((pending, m.get("content", "")))
            pending = None
    return out


def _count_lies(transcript, truth_fn):
    return sum(1 for q, a in _pairs(transcript)
               if truth_fn(q) in ("是", "否") and a.strip() != truth_fn(q))


# ---------------- ① 配额强制 ----------------
def test_quota_is_enforced_even_when_policy_always_lies():
    """政策永远选 lie，也恰好只骗 k 次——配额是分配问题不退化的最小约束。"""
    p = _puzzle()
    host = DeceptiveHost(policy=_AlwaysLiePolicy(), k=2)
    out = run_episode(p, host, _questioner(), max_rounds=MAX_ROUNDS,
                      train=False, train_host=True)
    assert out["n_lies"] == 2
    assert host.lies_told == 2 and host.budget_left == 0
    # 独立重算：逐轮比对答案与真值
    assert _count_lies(out["transcript"], _truth_fn(p)) == 2


def test_host_reset_restores_quota_between_games():
    """同一主持对象连打两局：第二局配额必须重置（run_episode 调 reset）。"""
    p = _puzzle()
    host = DeceptiveHost(policy=_AlwaysLiePolicy(), k=2)
    q = _questioner()
    o1 = run_episode(p, host, q, max_rounds=4, train=False, train_host=True)
    assert o1["n_lies"] == 2
    o2 = run_episode(p, host, q, max_rounds=4, train=False, train_host=True)
    assert o2["n_lies"] == 2, "第二局配额未重置 → 跨局泄漏"
    assert host.lies_told == 2, "lies_told 计数器未按局重置"


def test_reset_default_is_noop_for_existing_hosts():
    """向后兼容：不覆写 reset 的既有主持调用不炸、行为不变。"""
    CategoryHost().reset(_puzzle())
    MockHost().reset(_puzzle())


# ---------------- ② 状态题目无关 ----------------
def test_state_is_puzzle_independent():
    """同真相不同题 → 同状态键；且键内无题目身份（按题查表无从查起）。"""
    pa = Puzzle("A", "a", "面", "朋友与妻子")   # 人物关系=是
    pb = Puzzle("B", "b", "面", "同伴和姐姐")   # 人物关系=是（同伴/姐姐均在词表）
    pc = Puzzle("C", "c", "面", "热气球爆炸")   # 人物关系=否
    host = DeceptiveHost(k=2)
    q_text = render_question("人物关系")
    tr = [{"role": "question", "content": q_text}]
    ka = host.state_key(pa, q_text, tr)
    kb = host.state_key(pb, q_text, tr)
    kc = host.state_key(pc, q_text, tr)
    assert ka == kb, "同真相不同题状态不同 → 退化成按题查表，held-out 零迁移"
    assert ka != kc
    assert "A" not in ka and "B" not in ka
    parts = ka.split("|")
    assert len(parts) == 4 and parts[0] in ("是", "否")
    assert parts[1] == "1" and parts[2] == "2" and parts[3] == "0"
    # 复问后 asked_before 翻转（状态必须看得见复问，谎言才可被定位）
    tr2 = tr + [{"role": "answer", "content": "是"},
                {"role": "question", "content": q_text}]
    assert host.state_key(pa, q_text, tr2) == "是|2|2|1"


def test_undecidable_question_is_not_a_decision():
    """无法解析的问句：答「无关」，不构成决策，last_* 置空。"""
    p = _puzzle()
    host = DeceptiveHost(k=2)
    ans = host.answer(p, "今天天气怎么样", [])
    assert ans == "无关"
    assert host.last_state is None and host.last_action is None
    assert host.last_lied is False and host.budget_left == 2


# ---------------- ③ 奖励零和镜像 ----------------
def test_host_reward_is_zero_sum_mirror():
    """主持总奖励 = −提问者总奖励 − λ·谎言数（零和镜像 + 谎言成本，精确等式）。"""
    p = _puzzle()
    host = DeceptiveHost(policy=_AlwaysLiePolicy(), k=2, explore=False)
    out = run_episode(p, host, _questioner(), max_rounds=MAX_ROUNDS,
                      train=False, train_host=True)
    assert out["host_reward"] == pytest.approx(
        -out["reward"] - LIE_COST_DEFAULT * out["n_lies"])
    # 桩策略记录了 MC 回传的每次更新（终局镜像奖励 −2×F1 进入末步回报）
    assert host.policy.updates, "train_host=True 未做主持 Q 更新"


def test_terminal_mirror_sign():
    """终局镜像：提问者拿 +2×F1，主持拿 −2×F1（符号相反、量级相同）。

    用不重复提问者保证提问者总奖励为正（全有信息 +1×8 + 终局），
    此时主持总奖励必须为负——若镜像符号写反，本断言立即变红。
    """
    p = _puzzle()
    host = DeceptiveHost(policy=_AlwaysLiePolicy(), k=2, explore=False)
    q = _questioner(policy=_NoRepeatPolicy())
    out = run_episode(p, host, q, max_rounds=MAX_ROUNDS,
                      train=False, train_host=True)
    assert out["reward"] > 0, "测试前提失效：提问者总奖励非正"
    assert out["host_reward"] < 0
    # 撤掉谎言成本后等式仍精确成立（本测试用 λ=0 复算）
    out0 = run_episode(p, DeceptiveHost(policy=_AlwaysLiePolicy(), k=2,
                                        explore=False),
                       _questioner(policy=_NoRepeatPolicy()),
                       max_rounds=MAX_ROUNDS,
                       train=False, train_host=True, lie_cost=0.0)
    assert out0["host_reward"] == pytest.approx(-out0["reward"])


def test_train_host_loud_fails_on_non_marl_host():
    """train_host=True 但主持不暴露训练接口 → ValueError，不静默跳过。"""
    p = _puzzle()
    with pytest.raises(ValueError):
        run_episode(p, CategoryHost(), _questioner(), max_rounds=2,
                    train=False, train_host=True)


# ---------------- ④ tdhit 不被谎言刷高 ----------------
def test_tdhit_is_not_inflated_by_lies():
    """dhit 对「否→是」的谎言照单全收（表观命中不变），tdhit 不计。

    这正是 DESIGN §3.5 的核心：用 dhit 评欺骗环境会得出反向结论。
    """
    p = _puzzle()
    tf = _truth_fn(p)
    yes_cat = next(c for c in BANK if category_answer(p.solution, c) == "是")
    no_cat = next(c for c in BANK if category_answer(p.solution, c) == "否")

    def tr(pairs):
        out = []
        for q, a in pairs:
            out.append({"role": "question", "content": q})
            out.append({"role": "answer", "content": a})
        return out

    qy, qn = render_question(yes_cat), render_question(no_cat)
    truthful = tr([(qy, "是"), (qn, "否")])
    lying = tr([(qy, "否"), (qn, "是")])   # 两题都反转
    # 表观命中：谎言把「否」刷成「是」，dhit 照收
    assert distinct_hit_rate(lying) == distinct_hit_rate(truthful) == 0.5
    # 真值感知：谎言一律计不到
    assert tdhit(truthful, tf) == 1.0
    assert tdhit(lying, tf) == 0.0


def test_tdhit_denominator_is_total_rounds():
    """复问稀释：同一真值问题问 8 次，tdhit = 1/8（与 dhit 同口径）。"""
    p = _puzzle()
    tf = _truth_fn(p)
    yes_cat = next(c for c in BANK if category_answer(p.solution, c) == "是")
    q = render_question(yes_cat)
    tr = []
    for _ in range(8):
        tr.append({"role": "question", "content": q})
        tr.append({"role": "answer", "content": "是"})
    assert tdhit(tr, tf) == pytest.approx(1 / 8)


def test_tdhit_skips_undecidable_questions():
    """truth_fn 返回 None 的问句不进分子、仍占分母。"""
    tf = lambda q: None
    tr = [{"role": "question", "content": "闲聊"},
          {"role": "answer", "content": "是"}]
    assert tdhit(tr, tf) == 0.0
    assert tdhit([], tf) == 0.0


# ---------------- ⑤ 冲突丢弃恒等性 ----------------
def test_conflict_discard_is_identity_under_truthful_host():
    """诚实主持下同一类别永远同答 → 无冲突 → 还原句与旧行为逐位相同。"""
    p = _puzzle()
    q = _questioner()
    yes_cats = [c for c in BANK if category_answer(p.solution, c) == "是"][:3]
    tr = []
    for c in yes_cats:
        tr.append({"role": "question", "speaker": q.id,
                   "content": render_question(c)})
        tr.append({"role": "answer", "speaker": "host", "content": "是"})
    assert q.final_guess(p, tr) == (
        "真相与以下要素有关：" + "，".join(yes_cats) + "。")


def test_conflicting_category_is_discarded_whole():
    """冲突类别整体丢弃（抓谎）；未冲突类别不受影响。"""
    p = _puzzle()
    q = _questioner()
    yes_cats = [c for c in BANK if category_answer(p.solution, c) == "是"][:2]
    tr = [
        {"role": "question", "content": render_question(yes_cats[0])},
        {"role": "answer", "content": "是"},
        {"role": "question", "content": render_question(yes_cats[1])},
        {"role": "answer", "content": "是"},
        {"role": "question", "content": render_question(yes_cats[0])},
        {"role": "answer", "content": "否"},   # 复问翻转 → 冲突
    ]
    guess = q.final_guess(p, tr)
    assert yes_cats[0] not in guess, "冲突类别未被丢弃 → 抓谎机制失效"
    assert yes_cats[1] in guess


def test_guess_unchanged_under_truthful_host_across_seeds():
    """跨 5 种子集成验证：诚实主持下还原句 == 独立重算的确认类别集合。"""
    puzzles = load_puzzles()
    for seed in range(5):
        q = _questioner(policy=TabularQPolicy(N_CAT, seed=seed), explore=True)
        for p in puzzles[:3]:
            out = run_episode(p, CategoryHost(), q, max_rounds=MAX_ROUNDS,
                              train=False)
            confirmed, seen, pending = [], set(), None
            for m in out["transcript"]:
                if m["role"] == "question":
                    pending = m["content"]
                elif m["role"] == "answer" and pending is not None:
                    if m["content"] == "是":
                        kw = parse_question_keyword(pending)
                        if kw and kw not in seen:
                            seen.add(kw)
                            confirmed.append(kw)
                    pending = None
            expected = ("真相与以下要素有关：" + "，".join(confirmed) + "。"
                        if confirmed else "现有线索不足以还原真相。")
            assert out["guess"] == expected, (
                f"seed={seed} {p.soup_id}：诚实主持下还原句被改变 → "
                "冲突丢弃不是恒等变换，既有基线数字全部失效")


# ---------------- ⑥ run_episode 默认路径逐位不变 ----------------
def test_run_episode_default_path_unchanged():
    """默认参数（train_host=False）下：键集合精确为旧四个 + 数值独立重算复现。"""
    p = _puzzle()
    q = _questioner()
    out = run_episode(p, CategoryHost(), q, max_rounds=MAX_ROUNDS, train=False)
    assert set(out.keys()) == {"reward", "f1", "transcript", "guess"}
    roles = [m["role"] for m in out["transcript"]]
    assert roles.count("question") == MAX_ROUNDS
    assert roles.count("answer") == MAX_ROUNDS
    assert roles.count("guess") == 1
    # 独立重算奖励（复用模块自身 step/terminal 口径）
    asked, total, pending = [], 0.0, None
    for m in out["transcript"]:
        if m["role"] == "question":
            pending = m["content"]
        elif m["role"] == "answer" and pending is not None:
            kw = parse_question_keyword(pending)
            total += step_reward(kw in asked, m["content"])
            asked.append(kw)
            pending = None
    total += terminal_reward(out["guess"], p.solution)
    assert out["reward"] == pytest.approx(total)
    assert out["f1"] == pytest.approx(char_f1(out["guess"], p.solution))


# ---------------- 说谎基线 + 真实引擎贯通（E3 基础设施） ----------------
def test_random_liar_spends_exactly_k_lies():
    p = _puzzle()
    host = RandomLiarHost(max_rounds=MAX_ROUNDS, k=3, seed=0)
    host.reset(p)  # 谎言轮次在 reset 时抽样（引擎每局开头调用）
    assert len(host.lie_rounds) == 3
    out = run_episode(p, host, _questioner(), max_rounds=MAX_ROUNDS, train=False)
    assert _count_lies(out["transcript"], _truth_fn(p)) == 3
    assert host.lies_told == 3


def test_first_k_liar_lies_on_first_k_rounds_only():
    p = _puzzle()
    host = FirstKLiarHost(k=2)
    out = run_episode(p, host, _questioner(), max_rounds=MAX_ROUNDS, train=False)
    lied = [i + 1 for i, (q, a) in enumerate(_pairs(out["transcript"]))
            if a.strip() != _truth_fn(p)(q)]
    assert lied == [1, 2]


def test_marl_host_runs_inside_real_engine():
    """E3：DeceptiveHost 可直接进 TurtleSoupGame（Host 接口即插即用）。"""
    p = _puzzle()
    host = DeceptiveHost(k=2, seed=0)
    q = _questioner()
    game = TurtleSoupGame(p, host, [q], judge=MockJudge(), max_rounds=4)
    res = game.run()
    roles = [m["role"] for m in res.transcript]
    assert roles.count("question") == 4 and roles.count("answer") == 4
    assert roles.count("guess") == 1
    assert res.scores and res.scores[0]["score"] >= 0


def test_mixed_game_mock_questioner_vs_marl_host():
    """E3 混合对局：mock 提问者（关键词问句）× MARL 主持（golden_qa 真值）。

    真值源注入与 tdhit 的 truth_fn 同一思路——engine 不耦合语义模块。
    """
    p = _puzzle()

    def kw_truth(puzzle, question):
        for qa in puzzle.golden_qa:
            if qa.get("contains") and qa["contains"] in question:
                return qa.get("answer")
        return None

    host = DeceptiveHost(k=2, seed=0, truth_fn=kw_truth)
    game = TurtleSoupGame(p, host, [MockQuestioner(id="mock", seed=1)],
                          judge=MockJudge(), max_rounds=4)
    res = game.run()
    tr = list(res.transcript)  # GameResult 存的已是 dict
    assert sum(1 for m in tr if m["role"] == "answer") == 4
    # 有真值可反转的轮次才可能说谎；总谎言数 ≤ 配额
    tf = lambda q: kw_truth(p, q)
    assert _count_lies(tr, tf) <= 2


def test_invert_answer_roundtrip():
    assert invert_answer("是") == "否"
    assert invert_answer("否") == "是"


def test_category_truth_fn_matches_category_host():
    """默认真值源与 CategoryHost 完全同源（防两套判定漂移）。"""
    p = _puzzle()
    host = CategoryHost()
    for c in BANK:
        q_text = render_question(c)
        assert category_truth_fn(p, q_text) == host.answer(p, q_text, [])
    assert category_truth_fn(p, "闲聊") is None


# ---------------- ⑦ tdhit 恒等式（§6-A 发现 1：原 E1 判据因此弃用） ----------------
def test_per_lie_tdhit_damage_is_identity_one_over_rounds():
    """提问者不复问时，每谎 tdhit 伤害恒等于 1/max_rounds。

    DESIGN §6-A 发现 1 的证明：tdhit 同一问句只取首次答案（setdefault），
    谎言只发生在可决策问题上（如实回答本会是命中）。不复问时每次谎言
    恰好把一个本会命中的首答变成非命中 ⇒ per-lie damage = (L/R)/L = 1/R，
    **与谎言位置、配额 k、主持类型无关**。sanity 实测复问率恒 0.000
    （且发现 3 证明不复问理性），该区间即实验实际工作区。
    原 E1 判据（tdhit 每谎伤害 > random-liar）在此区间无法区分任何
    主持——不是负面结果，是仪表故障（判据已移至 F1，见 §6-A）。

    本测试把该退化固化为守护：若 tdhit 的命中/分母语义被无意改掉，
    恒等式破坏、本测试变红，提醒「E1' 已移至 F1」的依据依然成立。
    复问区间的吸收边界由 test_lie_on_repeat_is_absorbed_by_first_answer
    单独守护。
    """
    p = _puzzle()
    tf = _truth_fn(p)
    # 固定问句序列的提问者：问句选择与答案无关（RLQuestioner 状态只看
    # 已问动作），故诚实/说谎主持下问句序列逐位相同——恒等式的前提
    q = _questioner(policy=_NoRepeatPolicy())
    base_out = run_episode(p, CategoryHost(), q, max_rounds=MAX_ROUNDS,
                           train=False)
    base = tdhit(base_out["transcript"], tf)
    assert base == 1.0, "测试前提失效：诚实主持下 tdhit 应为 1.0"

    damages = []
    for k in (1, 2, 3):
        for seed in range(3):
            host = RandomLiarHost(max_rounds=MAX_ROUNDS, k=k, seed=seed)
            out = run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False)
            n_lies = _count_lies(out["transcript"], tf)
            assert n_lies >= 1, f"k={k} seed={seed}：该主持没说谎，无法验恒等式"
            damage = (base - tdhit(out["transcript"], tf)) / n_lies
            assert damage == pytest.approx(1.0 / MAX_ROUNDS, abs=1e-12), (
                f"k={k} seed={seed}：每谎 tdhit 伤害 {damage} ≠ 1/{MAX_ROUNDS} "
                "→ §6-A 发现 1 的恒等式被破坏，E1 判据依据失效")
            damages.append(damage)
    # 恒等式的要义：伤害与 k 无关（否则它还是个能区分主持的指标）
    assert max(damages) - min(damages) == pytest.approx(0.0, abs=1e-12)


def test_per_lie_tdhit_damage_identity_holds_for_trained_host_too():
    """恒等式对可学习主持同样成立（E1' 弃用 tdhit 不是因为主持类型）。"""
    p = _puzzle()
    tf = _truth_fn(p)
    q = _questioner(policy=_NoRepeatPolicy())
    base = tdhit(run_episode(p, CategoryHost(), q, max_rounds=MAX_ROUNDS,
                             train=False)["transcript"], tf)
    for k in (1, 2):
        host = DeceptiveHost(policy=_AlwaysLiePolicy(), k=k, explore=False)
        out = run_episode(p, host, q, max_rounds=MAX_ROUNDS,
                          train=False, train_host=True)
        n_lies = out["n_lies"]
        assert n_lies == k
        damage = (base - tdhit(out["transcript"], tf)) / n_lies
        assert damage == pytest.approx(1.0 / MAX_ROUNDS, abs=1e-12)


def test_lie_on_repeat_is_absorbed_by_first_answer():
    """复问区间的吸收边界（§6-A 发现 1 已知边界）：tdhit 同一问句只取
    首次答案（与 dhit 去重口径对齐的有意设计），故第二次起的谎言对
    tdhit 不可见——恒等式在此区间退化为 damage ≤ 1/R。

    固化该行为：它不是 bug（final_guess 的冲突丢弃看全部答案，抓谎机制
    不受影响），但意味着「主持倒是有个真实杠杆：谎言花在首问上」——
    只是判据方差来源变成提问者行为，仍不足以上升为 E1 主判据。
    """
    p = _puzzle()
    tf = _truth_fn(p)
    yes_cat = next(c for c in BANK if category_answer(p.solution, c) == "是")
    q = render_question(yes_cat)
    # 基线：同一问句问两次、两次都如实 → 首答去重后 1 命中 / 2 轮
    base_tr = [
        {"role": "question", "content": q},
        {"role": "answer", "content": "是"},
        {"role": "question", "content": q},
        {"role": "answer", "content": "是"},
    ]
    base = tdhit(base_tr, tf)
    assert base == pytest.approx(0.5)
    # 首次如实、复问撒谎 → 首答去重吸收该谎言，tdhit 不掉
    tr = [
        {"role": "question", "content": q},
        {"role": "answer", "content": "是"},
        {"role": "question", "content": q},
        {"role": "answer", "content": "否"},   # 复问谎言，被首答吸收
    ]
    assert tdhit(tr, tf) == pytest.approx(base)   # 吸收：伤害 0
    # 对照：首问就撒谎 → 首答即谎言，该问句的唯一命中被掉
    tr_first = [
        {"role": "question", "content": q},
        {"role": "answer", "content": "否"},
        {"role": "question", "content": q},
        {"role": "answer", "content": "是"},
    ]
    assert tdhit(tr_first, tf) == pytest.approx(0.0)   # 掉 1 命中
    # 同样是 1 谎：落点不同，每谎伤害 0 vs 1/R——主持的真实杠杆正在于此
    assert (base - tdhit(tr, tf)) / 1 == pytest.approx(0.0)
    assert (base - tdhit(tr_first, tf)) / 1 == pytest.approx(1.0 / 2)
    assert _count_lies(tr, tf) == 1 and _count_lies(tr_first, tf) == 1
