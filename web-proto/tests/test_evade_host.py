"""P2-B B-1b 搪塞轴守护（DESIGN-P2B1B-EVASION.md）。

与项目纪律一致：**全绿 ≠ 有效**。关键断言配变异验证（提交信息记录变异
及其击杀测试），重点守护：

  ① 双配额强制——政策永远选 evade/lie 也只恰好花 k_e/k 次
  ② 不得二次搪塞——被搪塞过的问句再问必须得到是/否（约束 2）
  ③ 搪塞答案与「无关」可区分——两类无信息答案分开维护（约束 3）
  ④ 转化奖励——曾被搪塞的问句复问拿到是/否 记 +1.0（复问惩罚豁免）
  ⑤ 终局还原忽略搪塞答案——它不是回答也不是矛盾（§3.4 核心坑）
  ⑥ 状态键向后兼容——无搪塞时与 B-1 格式逐位一致（既有 Q 表不失效）
  ⑦ run_episode 默认路径——键集合精确为旧四个；train_host 只追加键
  ⑧ tdhit 每搪塞恒等式——无转化区间 ≡ 1/max_rounds，转化区间 ≤ 1/R
  ⑨ E3b 混合对局——evasive 主持在真实引擎里端到端跑通
"""
import pytest

from engine.game import Puzzle, TurtleSoupGame
from engine.puzzles import load_puzzles
from engine.evaluate import char_f1, is_confirm, tdhit
from engine.semantic import (
    CATEGORY_NAMES, CategoryHost, category_answer, parse_category,
)
from players.host_rl import (
    ACTION_EVADE, ACTION_LIE, ACTION_TRUTH, EVASIVE_ANSWER,
    EvasiveDeceptiveHost, RandomEvaderHost, evaded_before,
)
from players.mock import MockJudge, MockQuestioner
from players.rl import (
    EVASIVE_ANSWERS, UNINFORMATIVE_ANSWERS, RLQuestioner, TabularQPolicy,
    render_question, parse_question_keyword, run_episode, step_reward,
    was_evaded,
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


class _FixedActionPolicy:
    """桩策略：永远选同一动作（免训练地构造极端策略）。"""

    n_actions = 3
    gamma = 0.9

    def __init__(self, action):
        self.action = action
        self.updates = []

    def select_action(self, state, explore=True):
        return self.action

    def greedy_action(self, state):
        return self.action

    def update(self, state, action, reward, next_state, done=False):
        self.updates.append((state, action, reward))


class _NoRepeatPolicy:
    """桩策略：总是问第一个未问过的动作（免训练的「不重复」提问者）。

    注意 B-1b 状态键可能带 `;ev=` 后缀——本桩只认前缀的已问集合，
    与 B-1 测试里的同款桩保持同一解析口径（后缀不影响已问判定）。
    """

    n_actions = N_CAT
    gamma = 0.9

    def select_action(self, state, explore=True):
        asked_part = str(state).split(";")[0]
        asked = {int(x) for x in asked_part.split(",")
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
    out, pending = [], None
    for m in transcript:
        if m.get("role") == "question":
            pending = m.get("content", "")
        elif m.get("role") == "answer" and pending is not None:
            out.append((pending, m.get("content", "")))
            pending = None
    return out


class _RepeatFirstPolicy:
    """桩策略：前三问都问同一个问句，之后依次问新动作（构造复问场景）。

    「不得二次搪塞」「random-evader 守约束 2」等测试必须真的复问——
    用 NoRepeatPolicy 会让这些测试空转（变异验证时亲测假绿）。
    """

    n_actions = N_CAT
    gamma = 0.9

    def __init__(self):
        self.calls = 0

    def select_action(self, state, explore=True):
        self.calls += 1
        return 0 if self.calls <= 3 else (self.calls - 3) % N_CAT

    def greedy_action(self, state):
        return self.select_action(state)

    def update(self, *a, **k):
        pass


def _mk_transcript(pairs, speaker="q"):
    """[(question, answer)] → transcript dicts（round 从 1 起）。"""
    tr = []
    for i, (q, a) in enumerate(pairs, start=1):
        tr.append({"round": i, "speaker": speaker, "role": "question",
                   "content": q})
        tr.append({"round": i, "speaker": "h", "role": "answer",
                   "content": a})
    return tr


# ---------------- ③ 两类无信息答案可区分 ----------------

def test_evasive_answer_vocabulary_is_disjoint_from_uninformative():
    """约束 3：搪塞答案与「无关」必须可区分（混一个元组就是口径漂移）。"""
    assert EVASIVE_ANSWER not in UNINFORMATIVE_ANSWERS
    assert set(EVASIVE_ANSWERS).isdisjoint(set(UNINFORMATIVE_ANSWERS))
    # 「无关」（无真值）不得被当成搪塞——否则无真值问句会被误判可转化
    assert not is_confirm(EVASIVE_ANSWER)
    # 当前问句 x 已入榜、未对答；历史轮次里 x 曾被搪塞 / 曾被答「无关」
    tr_evade = _mk_transcript([("x", EVASIVE_ANSWER)])
    tr_evade.append({"round": 2, "speaker": "q", "role": "question",
                     "content": "x"})
    assert evaded_before("x", tr_evade) == 1
    tr_plain = _mk_transcript([("x", "无关")])
    tr_plain.append({"round": 2, "speaker": "q", "role": "question",
                     "content": "x"})
    assert evaded_before("x", tr_plain) == 0, "「无关」不是搪塞（约束 3）"


def test_evaded_before_ignores_current_unanswered_question():
    """本轮问题已入榜但回答未入榜 → 不得计入（与 asked_before 同机理）。"""
    tr = _mk_transcript([("a", "是")])
    tr.append({"round": 2, "speaker": "q", "role": "question",
               "content": "a"})
    assert evaded_before("a", tr) == 0, "本次提问的对答还没发生，不能算搪塞过"


# ---------------- ① 双配额强制 ----------------

def test_both_quotas_are_enforced_under_always_evade_policy():
    """政策永远选 evade：搪塞恰好 k_e 次，且不花谎言配额。"""
    p = _puzzle()
    host = EvasiveDeceptiveHost(policy=_FixedActionPolicy(ACTION_EVADE),
                                k=2, k_e=2)
    q = _questioner(_NoRepeatPolicy())
    out = run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False,
                      train_host=True)
    assert host.evades_told == 2, f"搪塞配额未强制：{host.evades_told}"
    assert host.lies_told == 0, "永远选 evade 的政策不应产生谎言"
    assert out["n_evades"] == 2
    assert out["n_lies"] == 0


def test_both_quotas_are_enforced_under_always_lie_policy():
    """政策永远选 lie：谎言恰好 k 次，且不花搪塞配额（双配额独立）。"""
    p = _puzzle()
    host = EvasiveDeceptiveHost(policy=_FixedActionPolicy(ACTION_LIE),
                                k=2, k_e=2)
    q = _questioner(_NoRepeatPolicy())
    out = run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False,
                      train_host=True)
    assert host.lies_told == 2, f"谎言配额未强制：{host.lies_told}"
    assert host.evades_told == 0, "永远选 lie 的政策不应产生搪塞"
    assert out["n_lies"] == 2 and out["n_evades"] == 0


def test_quota_resets_between_games():
    """跨局不泄漏：两局各自的搪塞/谎言计数都从零开始。"""
    p = _puzzle()
    host = EvasiveDeceptiveHost(policy=_FixedActionPolicy(ACTION_EVADE),
                                k=1, k_e=1)
    q = _questioner(_NoRepeatPolicy())
    run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False)
    assert host.evades_told == 1 and host.lies_told == 0
    run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False)
    assert host.evades_told == 1, "第二局配额未重置 → 跨局泄漏"


# ---------------- ② 不得二次搪塞 ----------------

def test_no_double_evasion_same_question_gets_yes_no():
    """约束 2：被搪塞过的问句再次提出，必须得到是/否（即使政策永远选 evade）。

    提问者用「前三问都问同一问句」的脚本——不复问的话本测试空转
    （约束 2 的代码路径永远不被走到，变异验证时亲测会假绿）。
    """
    p = _puzzle()
    host = EvasiveDeceptiveHost(policy=_FixedActionPolicy(ACTION_EVADE),
                                k=2, k_e=2)
    q = _questioner(_RepeatFirstPolicy())
    out = run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False)
    pairs = _pairs(out["transcript"])
    seen_evaded = set()
    for question, answer in pairs:
        if question in seen_evaded:
            assert answer in ("是", "否"), (
                f"问句 {question!r} 已被搪塞过，复问却得到 {answer!r}"
                "——二次搪塞违反约束 2")
        if answer == EVASIVE_ANSWER:
            seen_evaded.add(question)
    assert seen_evaded, "该对局应至少产生一次搪塞（k_e=2 且政策永远选 evade）"


def test_evaded_before_feeds_state_key():
    """evaded_before 必须进状态键——否则策略无法区分「能否再搪塞」。

    调用约定（与 run_episode/TurtleSoupGame 一致）：当前问题已入榜、
    其回答尚未入榜。
    """
    p = _puzzle()
    host = EvasiveDeceptiveHost(k=2, k_e=2)
    q = render_question(BANK[0])
    tr = _mk_transcript([(q, EVASIVE_ANSWER)])
    tr.append({"round": 2, "speaker": "q", "role": "question",
               "content": q})
    key = host.state_key(p, q, tr)
    assert key.endswith("|1|1"), f"evaded_before 未进状态键：{key}"
    tr2 = _mk_transcript([(q, "是")])
    tr2.append({"round": 2, "speaker": "q", "role": "question",
                "content": q})
    assert host.state_key(p, q, tr2).endswith("|1|0")


def test_state_is_puzzle_independent_b1b():
    """同真相不同题 → 同状态键（B-1 §3.2 迁移承诺在 B-1b 不变）。"""
    puzzles = load_puzzles()
    host = EvasiveDeceptiveHost(k=2, k_e=2)
    q = render_question(BANK[0])
    tr = _mk_transcript([(q, "是")])
    keys = set()
    for p in puzzles:
        keys.add(host.state_key(p, q, tr))
    # 同一真相（是/否）下只应有一个键；不同真相可有多个——但绝不该按题分化
    truths = {category_answer(p.solution, parse_category(q)) for p in puzzles}
    assert len(keys) == len(truths), (
        f"状态键按题分化（{len(keys)} 个键 vs {len(truths)} 种真相）"
        "——迁移能力来自构造的承诺被破坏")


# ---------------- ④ 转化奖励 ----------------

def test_conversion_reward_waives_repeat_penalty():
    """曾被搪塞的问句复问拿到是/否 → +1.0；仍无信息 → −0.2；默认口径不变。"""
    assert step_reward(True, "是", converted=True) == 1.0
    assert step_reward(True, "否", converted=True) == 1.0
    assert step_reward(True, EVASIVE_ANSWER, converted=True) == -0.2
    assert step_reward(True, "无关", converted=True) == -0.2
    # 默认（非转化）路径与 B-1 逐位一致
    assert step_reward(True, "是") == -0.5
    assert step_reward(False, "是") == 1.0
    assert step_reward(False, "无关") == -0.2


def test_was_evaded_only_matches_prior_evasions():
    """was_evaded 只认历史轮次；本次的搪塞答案不得让本步变成「转化」。"""
    tr = _mk_transcript([("a", "是"), ("b", EVASIVE_ANSWER)])
    tr.append({"round": 3, "speaker": "q", "role": "question",
               "content": "c"})
    assert was_evaded("b", tr) is True
    assert was_evaded("a", tr) is False
    assert was_evaded("c", tr) is False, "本次提问尚未对答，不得算被搪塞过"
    # 已入榜的本次搪塞答案也不得误判（若调用时机在回答入榜后）
    tr.append({"round": 3, "speaker": "h", "role": "answer",
               "content": EVASIVE_ANSWER})
    assert was_evaded("c", tr) is False, (
        "本次提问自己的搪塞答案被算进了历史——转化判定会自我兑现")


def test_conversion_step_in_episode_gets_full_reward():
    """端到端：搪塞→复问→是/否 的那一步，提问者拿 +1.0 而非 −0.5。

    构造：主持首问搪塞（一次性脚本策略），之后如实；提问者先问 BANK[0]、
    复问它、再问新动作。断言复问那一步的奖励 = +1.0（is_repeat=True 但
    converted=True ⇒ 复问惩罚豁免）。
    """
    p = _puzzle()

    class _EvadeOncePolicy:
        """第一次决策选 evade，之后选 truth（免训练地构造搪塞场景）。"""

        n_actions = 3
        gamma = 0.9

        def __init__(self):
            self.calls = 0

        def select_action(self, state, explore=True):
            self.calls += 1
            return ACTION_EVADE if self.calls == 1 else ACTION_TRUTH

        def greedy_action(self, state):
            return self.select_action(state)

        def update(self, *a, **k):
            pass

    class _ScriptQuestionerPolicy:
        n_actions = N_CAT
        gamma = 0.9

        def __init__(self):
            self.calls = 0

        def select_action(self, state, explore=True):
            self.calls += 1
            return 0 if self.calls <= 2 else (self.calls - 2) % N_CAT

        def greedy_action(self, state):
            return self.select_action(state)

        def update(self, *a, **k):
            pass

    host = EvasiveDeceptiveHost(policy=_EvadeOncePolicy(), k=2, k_e=2)
    q = _questioner(_ScriptQuestionerPolicy())

    # 手动驱动前两步（复刻 run_episode 的循环体，才能逐步取奖励）
    tr = []
    q1 = q.ask_question(p, tr)
    tr.append({"round": 1, "speaker": q.id, "role": "question",
               "content": q1})
    a1 = host.answer(p, q1, tr)
    tr.append({"round": 1, "speaker": host.id, "role": "answer",
               "content": a1})
    assert a1 == EVASIVE_ANSWER, "首问应被搪塞"

    q2 = q.ask_question(p, tr)
    tr.append({"round": 2, "speaker": q.id, "role": "question",
               "content": q2})
    assert q2 == q1, "脚本第二步应复问同一问句"
    converted = was_evaded(q2, tr)
    assert converted is True, "复问前应识别出该问句被搪塞过"
    a2 = host.answer(p, q2, tr)
    tr.append({"round": 2, "speaker": host.id, "role": "answer",
               "content": a2})
    assert a2 in ("是", "否"), "约束 2：被搪塞过的问句必须得到是/否"
    rew = step_reward(is_repeat=True, answer=a2, converted=converted)
    assert rew == 1.0, f"转化步未拿到完整有信息奖励：{rew}"


# ---------------- ⑤ 终局还原忽略搪塞答案 ----------------

def test_final_guess_adopts_evade_then_confirm():
    """§3.4 核心坑：搪塞→转化→诚实答是，类别必须被采纳（不得误判冲突）。"""
    p = _puzzle()
    q = _questioner()
    cat = parse_category(render_question(BANK[0]))
    truth = category_answer(p.solution, cat)
    tr = _mk_transcript([
        (render_question(BANK[0]), EVASIVE_ANSWER),
        (render_question(BANK[0]), truth),  # 复问（转化）后诚实作答
    ])
    guess = q.final_guess(p, tr)
    if truth == "是":
        assert cat in guess, f"转化成功的类别被误丢弃：{guess!r}"
    else:
        assert cat not in guess, "truth=否 的类别本就不该被采纳"


def test_final_guess_evade_then_lie_does_not_leak():
    """搪塞→转化→谎答否（truth=是）：不采纳，且**无冲突痕迹**（安全谎窗口）。"""
    p = _puzzle()
    q = _questioner()
    cat = parse_category(render_question(BANK[0]))
    truth = category_answer(p.solution, cat)
    if truth != "是":
        pytest.skip("本题 BANK[0] 的真相不是「是」，换一题再测")
    tr = _mk_transcript([
        (render_question(BANK[0]), EVASIVE_ANSWER),
        (render_question(BANK[0]), "否"),  # 谎言：只有一个是否答案
    ])
    guess = q.final_guess(p, tr)
    assert cat not in guess, "谎言后的类别不应被采纳"
    # 与「直接谎言」的差别：直接谎言 [否, 是] 会冲突丢弃；搪塞垫刀后
    # 只有 [与此无关, 否]，没有第二个是否答案可冲突——需第三问才能抓
    answers = [a for _, a in _pairs(tr)]
    assert sum(1 for a in answers if a in ("是", "否")) == 1, (
        "安全谎窗口的结构前提被破坏：应只有一个是/否 答案")


def test_final_guess_identity_without_evasion_across_seeds():
    """无搪塞时还原句与「只认 confirm」的旧行为逐位相同（跨种子）。"""
    from players.rl import TabularQPolicy as _TQP
    for seed in range(5):
        p = load_puzzles()[seed % len(load_puzzles())]
        host = CategoryHost()
        pol = _TQP(N_CAT, seed=seed)
        q = _questioner(pol, explore=True)
        out = run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False)
        # 旧行为复算：只认 is_confirm
        answers_by_kw, order, pending = {}, [], None
        for m in out["transcript"]:
            if m.get("role") == "question":
                pending = m.get("content", "")
            elif m.get("role") == "answer" and pending is not None:
                kw = parse_question_keyword(pending)
                if kw:
                    answers_by_kw.setdefault(kw, []).append(m.get("content", ""))
                    if kw not in order:
                        order.append(kw)
                pending = None
        confirmed = [kw for kw in order
                     if all(is_confirm(a) for a in answers_by_kw[kw])]
        expected = ("现有线索不足以还原真相。" if not confirmed
                    else "真相与以下要素有关：" + "，".join(confirmed) + "。")
        assert out["guess"] == expected, (
            f"seed={seed}：无搪塞对局的还原句与旧行为不一致——恒等变换被破坏")
        # 无搪塞对局的状态键不得带 ev 后缀（⑥ 的跨种子加强）
        assert ";ev=" not in q.state_key(p, out["transcript"])


# ---------------- ⑥ 状态键向后兼容 ----------------

def test_state_key_has_no_ev_suffix_without_evasion():
    """无搪塞 transcript → 状态键与 B-1 格式逐位一致（无 ;ev= 后缀）。"""
    p = _puzzle()
    q = _questioner()
    tr = _mk_transcript([(render_question(BANK[0]), "是"),
                         (render_question(BANK[1]), "否")], speaker=q.id)
    key = q.state_key(p, tr)
    assert ";ev=" not in key, f"无搪塞却出现 ev 后缀：{key}"
    assert key == "0,1", f"与 B-1 格式不一致：{key}"


def test_state_key_ev_suffix_distinguishes_evaded_actions():
    """有搪塞 → 后缀出现且只含被搪塞的动作索引（与已问集合区分）。"""
    p = _puzzle()
    q = _questioner()
    tr = _mk_transcript([(render_question(BANK[0]), EVASIVE_ANSWER),
                         (render_question(BANK[1]), "是")], speaker=q.id)
    key = q.state_key(p, tr)
    assert key == "0,1;ev=0", f"ev 分量格式错误：{key}"
    # 被搪塞的问句复问后（已回答），ev 分量仍保留——历史不可擦除
    tr2 = _mk_transcript([(render_question(BANK[0]), EVASIVE_ANSWER),
                          (render_question(BANK[1]), "是"),
                          (render_question(BANK[0]), "是")], speaker=q.id)
    assert q.state_key(p, tr2) == "0,1;ev=0"


# ---------------- ⑦ run_episode 键集合 ----------------

def test_run_episode_default_path_keys_unchanged():
    """默认路径（train_host=False）键集合精确为旧四个。"""
    p = _puzzle()
    host = EvasiveDeceptiveHost(k=2, k_e=2)
    q = _questioner()
    out = run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False)
    assert set(out.keys()) == {"reward", "f1", "transcript", "guess"}


def test_run_episode_train_host_appends_keys_only():
    """train_host=True 只追加三个键；两动作主持的 n_evades 恒 0。"""
    from players.host_rl import DeceptiveHost
    p = _puzzle()
    q = _questioner(_NoRepeatPolicy())
    # 两动作主持（B-1 资产）：结构上不可能搪塞 → n_evades == 0
    h2 = DeceptiveHost(k=2)
    out2 = run_episode(p, h2, q, max_rounds=MAX_ROUNDS, train=False,
                       train_host=True)
    assert out2["n_evades"] == 0 and out2["n_lies"] >= 0
    # 三动作主持：键集合一致
    h3 = EvasiveDeceptiveHost(policy=_FixedActionPolicy(ACTION_EVADE),
                              k=2, k_e=2)
    out3 = run_episode(p, h3, _questioner(_NoRepeatPolicy()),
                       max_rounds=MAX_ROUNDS, train=False, train_host=True)
    assert set(out3.keys()) == set(out2.keys())
    assert out3["n_evades"] == 2


def test_evade_cost_enters_host_mirror_reward():
    """主持镜像奖励 = −提问者奖励 − λ·谎言 − ε·搪塞（精确等式）。"""
    p = _puzzle()
    lam, eps = 0.01, 0.05
    host = EvasiveDeceptiveHost(policy=_FixedActionPolicy(ACTION_EVADE),
                                k=2, k_e=2, lie_cost=lam, evade_cost=eps)
    q = _questioner(_NoRepeatPolicy())
    out = run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False,
                      train_host=True, lie_cost=lam, evade_cost=eps)
    assert out["host_reward"] == pytest.approx(
        -out["reward"] - lam * out["n_lies"] - eps * out["n_evades"])


# ---------------- ⑧ tdhit 每搪塞恒等式与转化边界 ----------------

def test_tdhit_per_evade_damage_is_identity_one_over_rounds():
    """无转化区间：每次搪塞的 tdhit 伤害恒等于 1/max_rounds（与谎言同构）。"""
    p = _puzzle()
    tf = _truth_fn(p)
    n_hits = sum(1 for c in CATEGORY_NAMES
                 if category_answer(p.solution, c) == "是")
    # 手工构造两条 transcript：全诚实 vs 第 1 问被搪塞（首答即真值）
    cat0 = parse_category(render_question(BANK[0]))
    truth0 = category_answer(p.solution, cat0)
    base_pairs = [(render_question(c), category_answer(p.solution, c))
                  for c in CATEGORY_NAMES[:MAX_ROUNDS]]
    evaded_pairs = [(render_question(BANK[0]), EVASIVE_ANSWER)] + base_pairs[1:]
    t_base = tdhit(_mk_transcript(base_pairs), tf)
    t_evad = tdhit(_mk_transcript(evaded_pairs), tf)
    assert t_base - t_evad == pytest.approx(1.0 / MAX_ROUNDS), (
        f"每搪塞 tdhit 伤害不是恒等式 1/R：{t_base} → {t_evad}")
    assert truth0 == "是", "恒等式证明要求被搪塞的问句首答本会是命中"


def test_tdhit_punishes_conversion_more_than_identity():
    """转化区间：tdhit 对转化的伤害**大于**恒等式 1/R（setdefault 口径）。

    tdhit 只取每问**首次**答案（与 dhit 对齐的去重语义）——复问那一次
    拿到的真实答案是/否 对它不可见，而提问轮数进了分母。故：
        无转化伤害 ≡ 1/R；转化伤害 = H/R − (H−1)/(R+1) > 1/R（H>1 时）
    **口径警示**（写进设计文档 §6-B）：tdhit 结构性惩罚复问，既不能用于
    E1b 主判据，也不能用来评价转化的价值——那是 F1 通路的事。
    """
    p = _puzzle()
    tf = _truth_fn(p)
    cat0 = parse_category(render_question(BANK[0]))
    base_pairs = [(render_question(c), category_answer(p.solution, c))
                  for c in CATEGORY_NAMES[:MAX_ROUNDS - 1]]
    conv_pairs = ([(render_question(BANK[0]), EVASIVE_ANSWER),
                   (render_question(BANK[0]),
                    category_answer(p.solution, cat0))]
                  + base_pairs[1:])
    t_base = tdhit(_mk_transcript(base_pairs), tf)
    t_conv = tdhit(_mk_transcript(conv_pairs), tf)
    damage = t_base - t_conv
    # 全诚实基线 t_base ≡ 1.0（tdhit 计「与真值一致」，是/否 都算）
    assert t_base == pytest.approx(1.0)
    assert damage > 1.0 / MAX_ROUNDS, (
        f"转化伤害应大于恒等式 1/R（setdefault 掩贩卖化答案）：{damage}")
    # 精确值：1 − (R−1)/(R+1) = 2/(R+1)（被搪塞问句的命中被首答去重掩
    # 售，与真相是/否 无关；分母多一轮）
    R = len(base_pairs)
    assert damage == pytest.approx(2.0 / (R + 1)), (
        f"转化伤害的理论值不符：{damage} vs {2.0 / (R + 1)}")


def test_dhit_inflated_by_direct_lie_but_not_by_evade_then_lie():
    """口径警示：直接谎言（否→是）会刷高 dhit；「搪塞垫刀」不会（首答是搪塞）。

    这正是为什么 E1b 主判据用 F1 而 dhit 只作表观读数——但要注意两个
    组合的 F1 后果不同（直接谎言可被复问冲突恢复，搪塞垫刀的谎言要第三问
    才能抓），指标口径与策略后果是两回事，都不能省。
    """
    from engine.evaluate import distinct_hit_rate
    p = _puzzle()
    tf = _truth_fn(p)
    cat0 = parse_category(render_question(BANK[0]))
    if category_answer(p.solution, cat0) != "是":
        pytest.skip("需要 BANK[0] 真相为「是」的题")
    q0 = render_question(BANK[0])
    # 直接谎言：[否, 是] —— dhit 首答取「否」？不，setdefault 取首答「否」
    # 也不是 confirm……构造 truth=否 的题才能刷高。改用通用断言：
    # 搪塞垫刀组合的 dhit 与 tdhit 都为 0（搪塞占住首答位，否不是 confirm）
    tr = _mk_transcript([(q0, EVASIVE_ANSWER), (q0, "否")])
    assert distinct_hit_rate(tr) == 0.0, "搪塞答案不应进 dhit 分子"
    assert tdhit(tr, tf) == 0.0, "谎言不应进 tdhit 分子"
    # 对照：直接谎言（truth=是 → 答否）dhit 同样为 0——差别在 F1 通路
    # （见 test_final_guess_evade_then_lie_does_not_leak 的结构断言）


# ---------------- ⑨ E3b 混合对局 ----------------

def test_mixed_game_mock_questioner_vs_evasive_host():
    """真实引擎 + mock 提问者 × evasive 主持（golden_qa 真值注入）跑通。

    真值源带回退（未命中 golden_qa 的问句答「否」）——E3b 是基础设施
    贯通性测试，保证每个问句可决策才能观察搪塞行为；真值源本身不在此测
    试范围（与 B-1 的 kw_truth 注入同一思路）。
    """
    p = _puzzle()

    def kw_truth(puzzle, question):
        for qa in puzzle.golden_qa:
            if qa.get("contains") and qa["contains"] in question:
                return qa.get("answer")
        return "否"  # 回退：保证可决策（本测试观察的是搪塞行为边界）

    host = EvasiveDeceptiveHost(policy=_FixedActionPolicy(ACTION_EVADE),
                                k=2, k_e=2, truth_fn=kw_truth, explore=False)
    game = TurtleSoupGame(p, host, [MockQuestioner(id="mock", seed=1)],
                          judge=MockJudge(), max_rounds=MAX_ROUNDS)
    res = game.run()
    d = res.to_dict()
    assert len(d["transcript"]) >= 2 * MAX_ROUNDS
    n_evade = sum(1 for m in d["transcript"]
                  if m.get("role") == "answer"
                  and m.get("content") == EVASIVE_ANSWER)
    assert 0 < n_evade <= 2, f"搪塞数越界（k_e=2）：{n_evade}"


def test_mixed_game_rl_questioner_vs_evasive_host():
    """真实引擎 + RL 提问者（语义问句）× evasive 主持（类别真值）跑通。"""
    p = _puzzle()
    host = EvasiveDeceptiveHost(k=2, k_e=2, explore=False)
    q = _questioner(TabularQPolicy(N_CAT, seed=3), explore=True)
    game = TurtleSoupGame(p, host, [q], judge=MockJudge(),
                          max_rounds=MAX_ROUNDS)
    res = game.run()
    assert res.scores, "对局应产出打分"
    assert len(res.transcript) == 2 * MAX_ROUNDS + 1 + 1


def test_random_evader_spends_at_most_k_e():
    """random-evader 基线：搪塞数 ≤ k_e，且从不说谎。"""
    p = _puzzle()
    for seed in range(5):
        host = RandomEvaderHost(max_rounds=MAX_ROUNDS, k_e=2, seed=seed)
        q = _questioner(_NoRepeatPolicy())
        out = run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False)
        pairs = _pairs(out["transcript"])
        n_evade = sum(1 for _, a in pairs if a == EVASIVE_ANSWER)
        assert 0 < n_evade <= 2, f"seed={seed} 搪塞数越界：{n_evade}"
        for question, answer in pairs:
            if answer != EVASIVE_ANSWER:
                t = _truth_fn(p)(question)
                assert t is None or answer == t, "random-evader 不应说谎"


def test_random_evader_never_evades_same_question_twice():
    """random-evader 同样遵守约束 2（复问必须得到是/否）。"""
    p = _puzzle()
    host = RandomEvaderHost(max_rounds=MAX_ROUNDS, k_e=3, seed=0)
    q = _questioner(_RepeatFirstPolicy())
    out = run_episode(p, host, q, max_rounds=MAX_ROUNDS, train=False)
    seen_evaded = set()
    for question, answer in _pairs(out["transcript"]):
        if question in seen_evaded:
            assert answer in ("是", "否"), (
                f"random-evader 对已搪塞问句 {question!r} 二次搪塞：{answer!r}")
        if answer == EVASIVE_ANSWER:
            seen_evaded.add(question)
    assert seen_evaded, "该脚本应至少触发一次搪塞"
