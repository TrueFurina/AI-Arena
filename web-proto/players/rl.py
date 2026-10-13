"""L2 执行层 · 强化学习提问策略（零成本可训练，接口预留 MARL 扩展）。

诚实定位（重要，勿夸大）：
  本模块同时承载**提问者侧 RL** 与 **MARL 主持训练的回路**（2026-10-13，
  P2-B B-1）：run_episode(train_host=True) 以零和镜像奖励同步训练主持侧
  策略（players/host_rl.py 的 DeceptiveHost），训练/评估共用同一循环。
  双方都是表格 Q 学习、无函数逼近——**不宣称 MARL 算法新颖性**，阶段目标
  是「评测台能评测 MARL agent」（见 DESIGN-P2B-MARL.md §1）。
  默认参数（train_host=False）下行为与单智能体时代逐位一致，有测试守护。

为什么动作空间必须是离散的：
  文本提问的动作空间无限，直接对文本做 RL 不可行。这里把「提问」抽象为
  「从有限候选库里选一条」，策略只在有限动作集上决策——这是让 RL 落到
  非结构化文本任务上的标准做法（动作抽象 / action abstraction）。

────────────────────────────────────────────────────────────────────
状态表示（2026-10-12 修正，这是本模块最关键的一次修复）
────────────────────────────────────────────────────────────────────
  state = 「题目 ID | 已问动作集合」

  早期版本 state 只有「已问动作集合」，不含题目 → 构成一个 POMDP：
  策略看不到当前是哪道题，却被要求区分「哪个动作对本题有用」。
  而动作库是由题库关键词派生的，天然与单题绑定（「肉」只在题001 有用、
  「热气球」只在题002 有用），于是同一状态 '' 在三个题上共享同一组 Q 值，
  数学上只能收敛到跨题混合平均，「偏好有信息量的动作」在这个表示下
  **根本不可表达**——不是学不出来，是表示不允许。

  把题目并入状态是**物理合法**的：真实海龟汤里，汤面（surface）本就是
  提问者可见的输入，不是答案泄漏。这里用 soup_id 作为题目键。

  ⚠️ 代价（必须一并陈述）：这样得到的策略是**按题查表的条件记忆**，
  对未训练过的新题零泛化（状态键不存在 → Q 全零 → 退化为随机）。
  要真正泛化，状态特征必须从「题目 ID」升级为「汤面的语义特征」，
  这是后续工作，当前**不宣称**具备泛化能力。

动作库（2026-10-12 改为从题库派生）：
  早期版本动作库是硬编码的 6 个词，存在一个致命漏洞：**题003 上没有任何
  一个动作是有效的**（其 golden_qa 关键词一个都没被收录）→ 三分之一
  的训练局里无论怎么问都只拿到惩罚，系统性污染所有动作的 Q 估计。
  现改为 build_action_bank() 从题库 golden_qa 自动派生，杜绝该漏洞。
    有信息型（答案为 是/否）——海龟汤里「否」同样缩小假设空间，计为有信息
    无信息型（答案为 无关）

实测可学信号（务必按事实陈述，勿照抄设计预期）：
  ① **不重复已问动作**——稳定可学，跨 5 个随机种子全部成立。
  ② **偏好有信息量的动作**——早期版本（无题目状态）**证伪**：
     eps×局数 5 组配置 + 每局随机抽题打散相位耦合，12 组中 8 组失败、
     排序随种子乱跳。根因即上述 POMDP 表示缺陷，非样本量问题。
     修正状态表示后已可学（见 tests/test_rl.py），但代价是退化为按题记忆。

────────────────────────────────────────────────────────────────────────────────
泛化能力实测（2026-10-12，held-out 协议，题库 12 题 / 动作库 92）
────────────────────────────────────────────────────────────────────────────────
  ⚠️ **结论是负的，且必须照实陈述**：本策略**不具备泛化能力**。

  puzzle（题目条件）模式，训练题数 3 / 6 / 9：
    训练题线索命中率 0.500 / 0.604 / 0.417
    新题线索命中率   0.111 / 0.167 / 0.333  ← 与未训练基线**逐位相同**
    真泛化增益       +0.000 / +0.000 / +0.000
  数学上是必然的：新题的状态键（soup_id|...）从未被训练过 → Q 全零 →
  行为退化为未训练策略。增益精确为 0，不是「接近 0」。

  ⚠️ 若不设未训练基线，会得出**假泛化**的错误结论：新题上策略一直问
  bank[0]（题库第一个关键词「海难」），而题001 恰好「海难=是」，
  于是新题命中率达 0.333，看起来「泛化得还行」。基线对照把它扣掉了。

  global（题目无关）模式：真泛化增益 **为负**（−0.056 / −0.042 / −0.125），
  且训练题命中率随训练题数增加反而下降（0.125 → 0.042 → 0.014）——
  与理论一致：训练题越多，跨题平均越均匀，Q 差异越小 → 趋近随机。

  样本阶梯 3→6→9：泛化增益**不随训练题增多而改善** → 这不是数据量问题，
  而是**动作抽象本身**的问题：动作是题目绑定的关键词，新题的关键词必然
  不同，没有任何迁移路径。唯一出路是把动作升级为**跨题通用的语义类别**
  （如「是否与人物关系有关」）。

  该改造已完成（2026-10-13，见 engine/semantic.py）：语义类别动作 + 题目
  无关状态，新题 dhit 从 0.00~0.17 升到 0.42~0.56，跨 5 种子全部优于关键词
  动作。但实测增益**小且不稳定**（扣除随机基线后 +0.04~+0.13，n_train=3/9
  时跨种子出现负值），且训练结果≈「按全局频率问」的参考值 → 只迁移到了
  全局先验。当前**仍不宣称具备泛化能力**，详见 semantic.py 的实测小节。

  评测协议见 eval_generalization.py，前提约束由 tests/test_generalization.py
  守护（其中「已知局限」类断言在真正实现泛化后应失败并予更新）。
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .base import Player
from engine.game import Puzzle
from engine.evaluate import char_f1, is_confirm  # is_confirm 走口径唯一真值

# 「无关」判定属 RL 奖励口径（无信息惩罚），与评测口径分离，故留在本模块
UNINFORMATIVE_ANSWERS = ("无关", "不重要")
# B-1b 搪塞答案词表（DESIGN-P2B1B-EVASION.md §2 约束 3）：与「无关」
# （不可解析、无真值）**分开维护**——提问者必须能区分「被搪塞（可转化）」
# 与「无真值（转化也无用）」，混进同一个元组就是口径漂移。
EVASIVE_ANSWERS = ("与此无关",)

REPEAT_PENALTY = -0.5
UNINFORMATIVE_PENALTY = -0.2
INFORMATIVE_REWARD = 1.0
TERMINAL_F1_WEIGHT = 2.0

POLICY_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "rl_policy.json",
)


def build_action_bank(puzzles: Iterable[Puzzle]) -> Tuple[List[str], Set[int]]:
    """从题库 golden_qa 派生动作库，返回 (动作列表, 有信息型动作的索引集合)。

    派生而非硬编码的理由：硬编码曾导致题003 上零有效动作（见模块文档）。
    分类口径：答案为「无关」= 无信息型；其余（是/否）= 有信息型。
    同一关键词若在两题上答案冲突，以「有信息」为准（保守，避免把有效
    动作误划为无效）。
    """
    informative: List[str] = []
    uninformative: List[str] = []
    for p in puzzles:
        for qa in p.golden_qa:
            kw = qa.get("contains")
            if not kw:
                continue
            if qa.get("answer") in UNINFORMATIVE_ANSWERS:
                if kw not in uninformative and kw not in informative:
                    uninformative.append(kw)
            else:
                if kw not in informative:
                    informative.append(kw)
                if kw in uninformative:
                    uninformative.remove(kw)
    bank = informative + uninformative
    return bank, set(range(len(informative)))


def render_question(keyword: str) -> str:
    """把动作（关键词）渲染成自然语言提问。

    渲染结果必须包含关键词本身：MockHost 按 golden_qa 的 contains 做子串
    匹配，关键词不在问句里就永远拿不到有效回答，RL 也就没有学习信号。
    """
    return f"这件事和「{keyword}」有关吗？"


def parse_question_keyword(question: str) -> Optional[str]:
    """从渲染后的问句里取回关键词（「…」之间的内容）。"""
    if "「" in question and "」" in question:
        return question.split("「", 1)[1].split("」", 1)[0]
    return None


def step_reward(is_repeat: bool, answer: str,
                converted: bool = False) -> float:
    """单步奖励：重复提问罚最重，无信息次之，有信息为正。

    「否」同样计为有信息：海龟汤里否定回答也缩小假设空间。

    converted=True（B-1b 转化）：本问句此前被搪塞、**从未得到过回答**——
    「复问刷哨」惩罚（−0.5）的立法目的是压制无新信息的重复提问，不适用
    于拿回一个从未得到过的答案；得到是/否 记 +1.0，仍是无信息则 −0.2。
    默认 False 时行为与 B-1 逐位一致（测试固化）。
    """
    if converted:
        if answer in UNINFORMATIVE_ANSWERS or answer in EVASIVE_ANSWERS:
            return UNINFORMATIVE_PENALTY
        return INFORMATIVE_REWARD
    if is_repeat:
        return REPEAT_PENALTY
    if answer in UNINFORMATIVE_ANSWERS or answer in EVASIVE_ANSWERS:
        return UNINFORMATIVE_PENALTY
    return INFORMATIVE_REWARD


def terminal_reward(guess: str, solution: str) -> float:
    """终局奖励：还原度（字符级 F1）加权。

    F1 由环境给出、用作奖励；策略本身**不观测** solution，不存在答案泄漏。
    """
    return TERMINAL_F1_WEIGHT * char_f1(guess, solution)


def was_evaded(question: str, transcript: Sequence[dict]) -> bool:
    """该问句在**本次提问之前**是否得到过搪塞答案（B-1b 转化判定）。

    口径：只扫**历史轮次**——transcript 末尾的最后一次提问（无论其对答
    是否已入榜）不计入。否则「本次自己的搪塞」会把本步误判成转化。
    该口径让函数对调用时机不敏感：run_episode 在回答入榜前调用，其它
    调用方在回答入榜后调用，结果一致。
    """
    q_idx = [i for i, m in enumerate(transcript)
             if m.get("role") == "question"]
    if not q_idx:
        return False
    boundary = q_idx[-1]  # 最后一次提问的位置：其后（含）不是历史轮次
    pending: Optional[str] = None
    for m in transcript[:boundary]:
        role = m.get("role")
        if role == "question":
            pending = m.get("content", "")
        elif role == "answer" and pending is not None:
            if pending == question and (m.get("content") or "") in EVASIVE_ANSWERS:
                return True
            pending = None
    return False


class TabularQPolicy:
    """离散动作空间的表格 Q 学习（epsilon-greedy）。

    state 接受两种形式：字符串（推荐，形如「题ID|已问动作」）或整数序列
    （旧格式，按已问动作集合编码）。保留旧格式仅为向后兼容，新代码一律
    走字符串状态——题目信息无法用动作索引表达。
    """

    def __init__(self, n_actions: int, alpha: float = 0.35, gamma: float = 0.9,
                 epsilon: float = 0.15, seed: int = 0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: Dict[str, List[float]] = {}
        self._rng = random.Random(seed)

    def _key(self, state) -> str:
        if isinstance(state, str):
            return state
        return ",".join(str(i) for i in sorted(set(state)))

    def _q(self, state) -> List[float]:
        k = self._key(state)
        if k not in self.Q:
            self.Q[k] = [0.0] * self.n_actions
        return self.Q[k]

    def select_action(self, state, explore: bool = True) -> int:
        """epsilon-greedy 选动作；explore=False 或 epsilon=0 时为纯贪心。"""
        q = self._q(state)
        if explore and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        # 贪心：并列时取 index 最小者（确定性，便于复现）
        return max(range(self.n_actions), key=lambda a: (q[a], -a))

    def greedy_action(self, state) -> int:
        return self.select_action(state, explore=False)

    def update(self, state, action: int, reward: float,
               next_state, done: bool = False) -> None:
        q = self._q(state)
        target = reward
        if not done:
            target += self.gamma * max(self._q(next_state))
        q[action] += self.alpha * (target - q[action])

    def save(self, path: str = POLICY_DEFAULT_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"n_actions": self.n_actions, "Q": self.Q}, f,
                      ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str = POLICY_DEFAULT_PATH, **kwargs) -> "TabularQPolicy":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        p = cls(n_actions=obj["n_actions"], **kwargs)
        p.Q = {k: list(v) for k, v in obj["Q"].items()}
        return p


class RLQuestioner(Player):
    """用 RL 策略选动作的提问者（实现 Player 接口，可直接进 TurtleSoupGame）。"""

    def __init__(self, policy: Optional[TabularQPolicy] = None,
                 id: str = "rl-questioner", model: str = "tabular-q",
                 action_bank: Optional[Sequence[str]] = None,
                 informative: Optional[Set[int]] = None,
                 explore: bool = True,
                 use_puzzle_state: bool = True):
        super().__init__(id, model)
        self.bank: List[str] = list(action_bank or [])
        if not self.bank:
            raise ValueError(
                "必须提供 action_bank（用 build_action_bank(puzzles) 从题库派生）："
                "硬编码动作库曾造成部分题目零有效动作的系统性漏洞。"
            )
        self.informative: Set[int] = set(informative or ())
        self.policy = policy or TabularQPolicy(len(self.bank))
        self.explore = explore
        # 关闭后回到早期（缺陷）表示，仅用于对照实验，勿用于生产
        self.use_puzzle_state = use_puzzle_state
        self.last_state: str = ""
        self.last_action: int = -1

    def _asked_actions(self, transcript: Sequence[dict]) -> List[int]:
        """从对局记录反推已问动作（无内部状态依赖，可安全用于任意引擎）。"""
        asked: List[int] = []
        for m in transcript:
            if m.get("role") != "question" or m.get("speaker") != self.id:
                continue
            kw = parse_question_keyword(m.get("content", ""))
            if kw in self.bank:
                asked.append(self.bank.index(kw))
        return asked

    def _evaded_actions(self, transcript: Sequence[dict]) -> List[int]:
        """被搪塞过的动作索引（B-1b）：问句得到过 EVASIVE_ANSWERS 回答。

        与 _asked_actions 同样从 transcript 反推（无内部状态），但需要
        问题→回答配对，故按 speaker 过滤后顺序扫描。多提问者对局下
        只认自己的问答对。
        """
        evaded: List[int] = []
        pending: Optional[str] = None
        for m in transcript:
            role = m.get("role")
            if role == "question":
                pending = (m.get("content", "")
                           if m.get("speaker") == self.id else None)
            elif role == "answer" and pending is not None:
                if (m.get("content") or "") in EVASIVE_ANSWERS:
                    kw = parse_question_keyword(pending)
                    if kw in self.bank:
                        idx = self.bank.index(kw)
                        if idx not in evaded:
                            evaded.append(idx)
                pending = None
        return evaded

    def state_key(self, puzzle: Puzzle, transcript: Sequence[dict]) -> str:
        asked = tuple(sorted(set(self._asked_actions(transcript))))
        prefix = f"{getattr(puzzle, 'soup_id', 'unknown')}|" if self.use_puzzle_state else ""
        # B-1b：ev 分量**仅在存在搪塞时追加**。空时格式与 B-1 逐位一致
        # ——无搪塞对局的状态键不变 ⇒ 既有 Q 表/策略文件/基线数字不受影响
        # （测试固化：同一 transcript 两种实现必须给出同一个键）。
        ev = tuple(sorted(set(self._evaded_actions(transcript))))
        suffix = f";ev={','.join(str(i) for i in ev)}" if ev else ""
        return f"{prefix}{','.join(str(i) for i in asked)}{suffix}"

    def ask_question(self, puzzle: Puzzle, transcript: List[dict]) -> str:
        state = self.state_key(puzzle, transcript)
        action = self.policy.select_action(state, explore=self.explore)
        self.last_state = state
        self.last_action = action
        return render_question(self.bank[action])

    def final_guess(self, puzzle: Puzzle, transcript: List[dict]) -> str:
        """终局还原：采纳被主持确认且**答案一贯**的线索，拼成紧凑陈述。

        早期版本把「问句原文 + 答案 + 模板废话」整段拼进去，而模板字
        （「RL 策略还原」「根据问答」「→」等）几乎不出现在汤底里，
        把精确率压到很低，导致 F1 对提问质量**不敏感**——策略学到的
        东西无法体现在还原质量上。现改为证据驱动：

          问中的线索越多 → 还原句命中的汤底字符越多 → F1 单调上升

        刻意不做两件事（诚实声明，勿为刷高数字而破例）：
          · 不回灌汤面原文——汤面与汤底字符重合度低，只会拉低精确率；
          · 不外推未经确认的情节——外推需要语义生成能力，当前动作抽象
            做不到，硬凑等于编造证据。
        因此 F1 上限受「关键词式短还原 vs 长汤底」的字符集合性质限制，
        实测 oracle（问中所有有效线索）约 0.3~0.4，见 tests/test_rl.py。

        P2-B 新增（DESIGN-P2B-MARL.md §3.4 冲突感知还原）：
          同一类别先后答案冲突（先是后否 / 先否后是）→ 该类别**整体丢弃**。
          这是提问者侧唯一的抓谎机制：复问同一类别，答案翻转即暴露。
          **向后兼容可证明**：诚实主持对同一类别永远同答，冲突不存在，
          该规则是恒等变换——现有全部基线数字不变（tests/test_marl_host.py
          固化为断言，含跨种子集成验证）。
        代价：复问 −0.5 且烧一轮，「花几轮验证 vs 多问一个新类别」成为
        提问者的真权衡（预注册失败模式②即观察这个权衡是否买贵了）。
        """
        answers_by_kw: Dict[str, List[str]] = {}
        order: List[str] = []
        pending: Optional[str] = None
        for m in transcript:
            role = m.get("role")
            if role == "question":
                pending = m.get("content", "")
            elif role == "answer" and pending is not None:
                kw = parse_question_keyword(pending)
                if kw:
                    if kw not in answers_by_kw:
                        answers_by_kw[kw] = []
                        order.append(kw)
                    answers_by_kw[kw].append(m.get("content", ""))
                pending = None
        # 一贯确认才采纳：全部回答均为确认 → 无冲突（诚实主持下恒成立）；
        # 出现确认与否定的冲突 → 该类别整体丢弃（抓谎）。
        # B-1b 修正：**搪塞答案在确认判定中被忽略**——它不是回答，也不是
        # 矛盾。不忽略的话，「搪塞→转化→诚实答是」会因那个与此无关被误判
        # 冲突而丢弃，转化永远无产出（设计文档 §3.4 的核心坑）。
        # 向后兼容：无搪塞的对局不存在搪塞答案，本规则是恒等变换
        # （tests/test_evade_host.py 跨种子固化）。
        confirmed: List[str] = [
            kw for kw in order
            if all(is_confirm(a) or (a or "") in EVASIVE_ANSWERS
                   for a in answers_by_kw[kw])
        ]
        if not confirmed:
            return "现有线索不足以还原真相。"
        return "真相与以下要素有关：" + "，".join(confirmed) + "。"


LIE_COST_DEFAULT = 0.05

# train_host=True 要求主持暴露的字段（DeceptiveHost 即如此）。缺字段则
# loud fail——静默跳过主持训练比报错危险（项目既有教训：静默 fallback）。
_HOST_TRAIN_INTERFACE = ("policy", "last_state", "last_action", "last_lied")


def run_episode(puzzle: Puzzle, host, questioner: RLQuestioner,
                max_rounds: int = 8, train: bool = True,
                train_host: bool = False,
                lie_cost: float = LIE_COST_DEFAULT,
                evade_cost: float = 0.0) -> dict:
    """跑一局（RL 回路），返回 {reward, f1, transcript, guess}。

    train=True 时做提问者 Q 更新。训练与评估共用本函数，避免两套循环实现漂移。
    提问走 questioner.ask_question（而非直接调 policy），确保 Player 接口
    与真实引擎 TurtleSoupGame 的行为完全一致。

    max_rounds 默认 8：默认 4 时即便每轮都问中有效线索，还原句也只有
    ~9 个去重字符，对 30~70 字的汤底召回率约 0.1，F1 被指标本身压死，
    看不出策略差异。

    train_host=True（P2-B MARL，见 DESIGN-P2B-MARL.md §3.3/§4）：同步训练
    主持侧策略，零和镜像——主持每步奖励 = −提问者当步奖励 − λ·[lie]，
    终局奖励 = −2×F1。要求 host 暴露 policy/last_state/last_action/last_lied
    （DeceptiveHost），否则 ValueError。该参数默认 False，且返回 dict 只在
    开启时**追加** host_reward/n_lies/n_evades 三个键——默认路径的键集合
    与数值逐位不变（tests/test_marl_host.py 固化守护）。

    evade_cost（B-1b，DESIGN-P2B1B-EVASION.md §3.3）：每次搪塞的固定成本，
    默认 0。主持侧读 `last_evaded`（getattr 默认 False——两动作主持
    结构上不可能搪塞，这不是静默跳过而是能力缺失的结构性事实）；
    EvasiveDeceptiveHost 暴露该字段。转化判定（was_evaded）在主持回答
    入榜前计算，只 matched 历史轮次的搪塞答案。

    每局开头调用 host.reset(puzzle)（hasattr 守卫）：DeceptiveHost 的
    谎言配额等每局状态在此归零；无 reset 的主持行为零变化。
    """
    if hasattr(host, "reset"):
        host.reset(puzzle)
    if train_host:
        missing = [a for a in _HOST_TRAIN_INTERFACE if not hasattr(host, a)]
        if missing:
            raise ValueError(
                f"train_host=True 需要主持暴露 {missing}（DeceptiveHost）；"
                "拒绝静默跳过主持训练"
            )

    transcript: List[dict] = []
    asked: List[int] = []
    total = 0.0
    traj: List[Tuple[str, int, float]] = []
    host_traj: List[Tuple[str, int, float]] = []
    host_total = 0.0
    n_lies = 0
    n_evades = 0

    for r in range(1, max_rounds + 1):
        question = questioner.ask_question(puzzle, transcript)
        action = questioner.last_action
        state = questioner.last_state
        transcript.append({"round": r, "speaker": questioner.id,
                           "role": "question", "content": question})

        # B-1b 转化判定：必须在主持回答入榜前——此时本次提问尚无对答，
        # 只会 matched 到历史轮次的搪塞（was_evaded 文档）。
        converted = was_evaded(question, transcript)

        answer = host.answer(puzzle, question, transcript)
        transcript.append({"round": r, "speaker": getattr(host, "id", "host"),
                           "role": "answer", "content": answer})

        is_repeat = action in asked
        rew = step_reward(is_repeat, answer, converted=converted)
        asked.append(action)
        traj.append((state, action, rew))
        total += rew

        if train_host:
            h_state = host.last_state
            h_action = host.last_action
            if h_state is None or h_action is None:
                continue  # 本轮主持无决策（问题无法解析），不进主持轨迹
            h_lied = bool(host.last_lied)
            h_evaded = bool(getattr(host, "last_evaded", False))
            h_rew = (-rew - (lie_cost if h_lied else 0.0)
                     - (evade_cost if h_evaded else 0.0))
            host_traj.append((h_state, h_action, h_rew))
            host_total += h_rew
            if h_lied:
                n_lies += 1
            if h_evaded:
                n_evades += 1

    guess = questioner.final_guess(puzzle, transcript)
    transcript.append({"round": max_rounds + 1, "speaker": questioner.id,
                       "role": "guess", "content": guess})
    f1 = char_f1(guess, puzzle.solution)
    term = terminal_reward(guess, puzzle.solution)
    total += term

    if train:
        # 蒙特卡洛回报回传（信用分配修正）。
        # 早期版本把终局奖励只记在最后一个动作上，造成明显偏差：碰巧排在末位的
        # 无关动作会被终局正奖励抬高——实测无关动作「钱」Q 被抬到 +0.457，
        # 反而高于部分信息型动作。改为从后往前按 MC 回报回传给每一步。
        G = term
        for (state, action, rew) in reversed(traj):
            G = rew + questioner.policy.gamma * G
            questioner.policy.update(state, action, G, "|", done=True)

    if train_host:
        # 主持侧同一套 MC 回传；终局镜像奖励 = −2×F1（零和）
        G = -term
        for (state, action, rew) in reversed(host_traj):
            G = rew + host.policy.gamma * G
            host.policy.update(state, action, G, "|", done=True)
        host_total += -term

    out = {"reward": total, "f1": f1, "transcript": transcript, "guess": guess}
    if train_host:
        out["host_reward"] = host_total
        out["n_lies"] = n_lies
        out["n_evades"] = n_evades
    return out
