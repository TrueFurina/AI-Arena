"""P2-B MARL 主持侧：可学习欺骗策略 + 两个说谎基线（限量欺骗 · 零和自对弈）。

设计依据 DESIGN-P2B-MARL.md §3：照实主持的决策空间为零（答案由汤底机械
派生），可学习主持必须有真实决策，否则实验在设计上就是空的。因此引入
项目的「欺骗」维度：主持可在诚实/反转答案间选择，受每局谎言配额 k 约束
（无配额时「处处说谎」是无成本占优策略，分配问题退化）。

本模块主持共享同一套「真值→回答」管路，差别只在**谁决定哪一轮做什么**：

- DeceptiveHost  表格 Q 学习（状态题目无关，迁移能力来自构造而非祈祷）
- RandomLiarHost 基线：同配额 k，谎言位置均匀随机（E1 的关键对照）
- FirstKLiarHost 基线：前 k 轮全说谎（退化策略对照）
- EvasiveDeceptiveHost（B-1b，DESIGN-P2B1B-EVASION.md）三动作 + 双配额
- RandomEvaderHost（B-1b 基线）同 k_e，搪塞位置均匀随机（E1b 对照）

真值来源可注入（truth_fn，与 engine/evaluate.tdhit 的同名参数同一思路）：
默认按语义类别从汤底派生（MARL 设置，与 CategoryHost 同源，非新判据）；
E3 混合对局可注入关键词（golden_qa）真值，让 mock 提问者的问句也有
可反转的真值。

边界（诚实声明，勿夸大）：
- 主持看得见汤底——物理合法（Host 本来就知道汤底），提问者看不见；
- 状态刻意题目无关（§3.2）：按题查表的主持策略在 held-out 上零迁移；
- 本模块不自更新 Q：奖励计算与 MC 回传集中在 players/rl.py 的
  run_episode(train_host=True)，与提问者侧完全对称，避免两套学习回路漂移。
"""
from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Set

from .base import Host
from engine.game import Puzzle
from engine.semantic import category_answer, parse_category
from players.rl import EVASIVE_ANSWERS, TabularQPolicy

# 动作编号（与 Q 表列对应）：0=如实 / 1=说谎
ACTION_TRUTH = 0
ACTION_LIE = 1
# B-1b 第三动作：搪塞（DESIGN-P2B1B-EVASION.md §3.1）
ACTION_EVADE = 2

# 搪塞答案（词表唯一真值在 players/rl.py 的 EVASIVE_ANSWERS——与「无关」
# 分开维护的理由见该处注释；这里取首元素供主持返回）。
EVASIVE_ANSWER = EVASIVE_ANSWERS[0]

LIE_COST_DEFAULT = 0.05


def invert_answer(ans: str) -> str:
    """是↔否 反转（调用方保证入参是 是/否，见各 host 的 truth_of 守卫）。"""
    return "否" if ans == "是" else "是"


def category_truth_fn(puzzle: Puzzle, question: str) -> Optional[str]:
    """默认真值源：语义类别 → 从汤底文本派生（与 CategoryHost 完全同源）。"""
    cat = parse_category(question)
    if cat is None:
        return None
    return category_answer(puzzle.solution, cat)


def current_round(transcript: Sequence[dict]) -> int:
    """当前轮次 = 已记录的问题数。

    前提：answer() 被调用时，本轮问题已入 transcript（run_episode 与
    TurtleSoupGame 都如此）。多提问者对局下计数是全局问题数——MARL 设置
    为单提问者，E3 混合对局亦为单提问者。
    """
    return sum(1 for m in transcript if m.get("role") == "question")


def asked_before(question: str, transcript: Sequence[dict]) -> int:
    """该问句此前是否出现过（本轮问题已入 transcript，故 >1 即复问）。"""
    n = sum(1 for m in transcript
            if m.get("role") == "question" and m.get("content") == question)
    return 1 if n > 1 else 0


def evaded_before(question: str, transcript: Sequence[dict]) -> int:
    """该问句此前是否被搪塞过（B-1b，与 asked_before 对位）。

    只数**搪塞答案**（EVASIVE_ANSWERS）：不可决策问题的「无关」不是
    搪塞——两类答案必须可区分，否则「被搪塞过 → 必须给是/否」的约束
    （DESIGN-P2B1B §2 约束 2）会误伤无真值问句。
    口径同 players/rl.py 的 was_evaded：transcript 末尾的最后一次提问
    （本轮）不计入，只扫历史轮次——调用时机（回答是否已入榜）不敏感。
    """
    q_idx = [i for i, m in enumerate(transcript)
             if m.get("role") == "question"]
    if not q_idx:
        return 0
    boundary = q_idx[-1]
    n = 0
    pending: Optional[str] = None
    for m in transcript[:boundary]:
        role = m.get("role")
        if role == "question":
            pending = m.get("content")
        elif role == "answer" and pending is not None:
            if (pending == question
                    and (m.get("content") or "") in EVASIVE_ANSWERS):
                n += 1
            pending = None
    return 1 if n > 0 else 0


class _SoupHostBase(Host):
    """三个说谎主持共享的管路：真值解析 + 反转 + 轮次/复问判定。"""

    def __init__(self, id: str, model: str,
                 truth_fn: Optional[Callable[[Puzzle, str], Optional[str]]] = None):
        super().__init__(id=id, model=model)
        self._truth_fn = truth_fn or category_truth_fn
        self.lies_told = 0

    def truth_of(self, puzzle: Puzzle, question: str) -> Optional[str]:
        """如实答案。None 或非二元值（如「无关」）= 无真值可反转。"""
        return self._truth_fn(puzzle, question)

    def _decidable_truth(self, puzzle: Puzzle, question: str) -> Optional[str]:
        """只有 是/否 才算可决策真值——「无关」等问题不构成谎言机会。"""
        t = self.truth_of(puzzle, question)
        return t if t in ("是", "否") else None

    def reset(self, puzzle) -> None:
        """每局重置（Host ABC 默认 no-op；有每局计数器的主持覆写）。"""
        self.lies_told = 0


class DeceptiveHost(_SoupHostBase):
    """可学习欺骗主持：表格 Q + 谎言配额 k + 零和镜像奖励（§3.2/§3.3）。

    状态（题目无关）：
        state = f"{truth}|{round}|{budget_left}|{asked_before}"
        truth         当前问题的如实答案（是/否）——主持本来就知道汤底
        round         第几轮（由 transcript 问题数推得，不需要 max_rounds）
        budget_left   剩余谎言配额（k..0）
        asked_before  该问句此前是否问过（0/1）
    动作：0=如实 / 1=说谎。配额耗尽时政策选 1 也强制执行 0（配额强制），
    且 Q 更新记录的是**执行**的动作（on-policy）——用未执行动作的回报
    去更新它等于把探索噪声写进策略。

    训练/评估开关：explore=True 走 epsilon-greedy（训练或 §4 的 ε=0.1
    随机化冻结）；explore=False 纯贪心（评估）。
    """

    def __init__(self, policy: Optional[TabularQPolicy] = None,
                 k: int = 2, lie_cost: float = LIE_COST_DEFAULT,
                 epsilon: float = 0.15, seed: int = 0, explore: bool = True,
                 id: str = "deceptive-host", truth_fn=None):
        super().__init__(id=id, model="deceptive-tabular-q", truth_fn=truth_fn)
        self.policy = policy or TabularQPolicy(2, epsilon=epsilon, seed=seed)
        self.k = k
        self.lie_cost = lie_cost
        self.explore = explore
        self.budget_left = k
        # run_episode(train_host=True) 读取这三个字段（模式与 RLQuestioner 的
        # last_state/last_action 一致）；无决策轮次置 None，调用方据此跳过。
        self.last_state: Optional[str] = None
        self.last_action: Optional[int] = None
        self.last_lied: bool = False

    def reset(self, puzzle) -> None:
        self.budget_left = self.k
        self.lies_told = 0
        self.last_state = None
        self.last_action = None
        self.last_lied = False

    def state_key(self, puzzle: Puzzle, question: str,
                  transcript: Sequence[dict]) -> str:
        """当前状态键（仅对有真值的问题有意义；无法解析时 answer 不走此路）。

        单独成方法是为了可测：「同真相不同题 → 同状态」是「迁移能力来自
        构造」的核心承诺，必须能直接断言，而不是只能从训练副产品反推。
        """
        truth = self._decidable_truth(puzzle, question)
        return (f"{truth}|{current_round(transcript)}|{self.budget_left}"
                f"|{asked_before(question, transcript)}")

    def answer(self, puzzle: Puzzle, question: str, transcript: List[dict]) -> str:
        truth = self._decidable_truth(puzzle, question)
        if truth is None:
            # 无真值可反转：答「无关」，不构成决策，不进训练轨迹
            self.last_state = None
            self.last_action = None
            self.last_lied = False
            return "无关"
        state = self.state_key(puzzle, question, transcript)
        action = self.policy.select_action(state, explore=self.explore)
        lie = action == ACTION_LIE and self.budget_left > 0
        if lie:
            self.budget_left -= 1
            self.lies_told += 1
        self.last_state = state
        self.last_action = ACTION_LIE if lie else ACTION_TRUTH
        self.last_lied = lie
        return invert_answer(truth) if lie else truth


class RandomLiarHost(_SoupHostBase):
    """基线：同配额 k，谎言位置均匀随机（E1 的关键对照）。

    开局（reset）从 1..max_rounds 均匀抽 k 个不同轮次，恰好撒 k 个谎。
    与 DeceptiveHost 的差别只在「谁说谎由学出来的策略决定」——控制变量。
    """

    def __init__(self, max_rounds: int = 8, k: int = 2, seed: int = 0,
                 id: str = "random-liar", truth_fn=None):
        super().__init__(id=id, model="random-liar", truth_fn=truth_fn)
        self.max_rounds = max_rounds
        self.k = k
        self._rng = random.Random(seed)
        self.lie_rounds: Set[int] = set()

    def reset(self, puzzle) -> None:
        n = min(self.k, self.max_rounds)
        self.lie_rounds = set(self._rng.sample(range(1, self.max_rounds + 1), n))
        self.lies_told = 0

    def answer(self, puzzle: Puzzle, question: str, transcript: List[dict]) -> str:
        truth = self._decidable_truth(puzzle, question)
        if truth is None:
            return "无关"
        if current_round(transcript) in self.lie_rounds:
            self.lies_told += 1
            return invert_answer(truth)
        return truth


class FirstKLiarHost(_SoupHostBase):
    """基线：前 k 轮全说谎（退化策略对照——不挑位置，照面就骗）。"""

    def __init__(self, k: int = 2, id: str = "first-k-liar", truth_fn=None):
        super().__init__(id=id, model="first-k-liar", truth_fn=truth_fn)
        self.k = k

    def answer(self, puzzle: Puzzle, question: str, transcript: List[dict]) -> str:
        truth = self._decidable_truth(puzzle, question)
        if truth is None:
            return "无关"
        if current_round(transcript) <= self.k:
            self.lies_told += 1
            return invert_answer(truth)
        return truth


# ---------------------------------------------------------------------------
# B-1b 搪塞轴（DESIGN-P2B1B-EVASION.md）。B-1 的三个类 above 一字未动——
# 负结果是已发布资产；新轴用新类实现，参照列（trained-deceptive /
# random-liar）保证与 B-1 同口径可比。
# ---------------------------------------------------------------------------

class EvasiveDeceptiveHost(_SoupHostBase):
    """B-1b 可学习搪塞主持：三动作 + 双配额（设计文档 §3.1/§3.2）。

    动作 0=如实 / 1=说谎 / 2=搪塞；谎言配额 k 与搪塞配额 k_e **独立**。
    两条硬约束：
      ① 同一问句不得二次搪塞（evaded_before=1 时选 evade 也强制执行
         truth）——被搪塞过的问句必须得到是/否，搪塞因此是**延迟**工具
         而非否认工具，提问者保有转化权；
      ② 配额耗尽的 infeasible 动作强制执行 truth，且 Q 更新记录**执行**的
         动作（on-policy，与 DeceptiveHost 同一纪律）。
    无真值问句答「无关」，不构成决策，不进训练轨迹（与 B-1 一致）。

    状态（六维，题目无关——B-1 §3.2 的迁移承诺不变）：
        state = f"{truth}|{round}|{lie_left}|{evade_left}|{asked_before}|{evaded_before}"
    """

    def __init__(self, policy: Optional[TabularQPolicy] = None,
                 k: int = 2, k_e: int = 2,
                 lie_cost: float = LIE_COST_DEFAULT, evade_cost: float = 0.0,
                 epsilon: float = 0.15, seed: int = 0, explore: bool = True,
                 id: str = "evasive-host", truth_fn=None):
        super().__init__(id=id, model="evasive-tabular-q", truth_fn=truth_fn)
        self.policy = policy or TabularQPolicy(3, epsilon=epsilon, seed=seed)
        self.k = k
        self.k_e = k_e
        self.lie_cost = lie_cost
        self.evade_cost = evade_cost
        self.explore = explore
        self.budget_left = k
        self.evade_left = k_e
        # run_episode(train_host=True) 读取的字段（模式与 DeceptiveHost
        # 一致，多一个 last_evaded）；无决策轮次置 None。
        self.last_state: Optional[str] = None
        self.last_action: Optional[int] = None
        self.last_lied: bool = False
        self.last_evaded: bool = False
        self.evades_told = 0

    def reset(self, puzzle) -> None:
        self.budget_left = self.k
        self.evade_left = self.k_e
        self.lies_told = 0
        self.evades_told = 0
        self.last_state = None
        self.last_action = None
        self.last_lied = False
        self.last_evaded = False

    def state_key(self, puzzle: Puzzle, question: str,
                  transcript: Sequence[dict]) -> str:
        truth = self._decidable_truth(puzzle, question)
        return (f"{truth}|{current_round(transcript)}|{self.budget_left}"
                f"|{self.evade_left}|{asked_before(question, transcript)}"
                f"|{evaded_before(question, transcript)}")

    def answer(self, puzzle: Puzzle, question: str, transcript: List[dict]) -> str:
        truth = self._decidable_truth(puzzle, question)
        if truth is None:
            self.last_state = None
            self.last_action = None
            self.last_lied = False
            self.last_evaded = False
            return "无关"
        state = self.state_key(puzzle, question, transcript)
        action = self.policy.select_action(state, explore=self.explore)
        # 约束 ①②：infeasible → truth（不静默改判成另一种欺骗动作）
        if action == ACTION_LIE and self.budget_left <= 0:
            action = ACTION_TRUTH
        if action == ACTION_EVADE and (self.evade_left <= 0
                                       or evaded_before(question, transcript)):
            action = ACTION_TRUTH
        if action == ACTION_LIE:
            self.budget_left -= 1
            self.lies_told += 1
        elif action == ACTION_EVADE:
            self.evade_left -= 1
            self.evades_told += 1
        self.last_state = state
        self.last_action = action
        self.last_lied = action == ACTION_LIE
        self.last_evaded = action == ACTION_EVADE
        if action == ACTION_LIE:
            return invert_answer(truth)
        if action == ACTION_EVADE:
            return EVASIVE_ANSWER
        return truth


class RandomEvaderHost(_SoupHostBase):
    """B-1b 基线：同 k_e，搪塞位置均匀随机（E1b 的关键对照）。

    开局（reset）从 1..max_rounds 均匀抽 k_e 个不同轮次；轮到该轮次、
    问题可决策且未被搪塞过 → 搪塞，否则如实答。**不说谎**——与
    random-liar 对位，控制变量是「搪塞位置由学出的策略决定」。
    与 RandomLiarHost 同构：抽中的轮次若撞上不可决策问题则该次作废
    （B-1 已有同款行为，保持对照公平）。
    """

    def __init__(self, max_rounds: int = 8, k_e: int = 2, seed: int = 0,
                 id: str = "random-evader", truth_fn=None):
        super().__init__(id=id, model="random-evader", truth_fn=truth_fn)
        self.max_rounds = max_rounds
        self.k_e = k_e
        self._rng = random.Random(seed)
        self.evade_rounds: Set[int] = set()
        self.evades_told = 0

    def reset(self, puzzle) -> None:
        n = min(self.k_e, self.max_rounds)
        self.evade_rounds = set(self._rng.sample(range(1, self.max_rounds + 1), n))
        self.lies_told = 0
        self.evades_told = 0

    def answer(self, puzzle: Puzzle, question: str, transcript: List[dict]) -> str:
        truth = self._decidable_truth(puzzle, question)
        if truth is None:
            return "无关"
        if (current_round(transcript) in self.evade_rounds
                and not evaded_before(question, transcript)):
            self.evades_told += 1
            return EVASIVE_ANSWER
        return truth
