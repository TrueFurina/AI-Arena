"""L2 执行层 · 强化学习提问策略（零成本可训练，接口预留 MARL 扩展）。

诚实定位（重要，勿夸大）：
  当前实现是**单智能体 RL**——只学习提问者侧的「动作选择」策略。
  对局环境本身是多智能体（主持 + 提问者），但主持当前是固定策略（Mock），
  不参与学习。完整 MARL（提问者与主持联合训练 / 自对弈对抗欺骗）为后续
  扩展；本模块 Policy 接口（select_action / update）已按可直接扩展设计，
  联合训练只需再实例化一个策略给主持并共用同一 reward 通道。

为什么动作空间必须是离散的：
  文本提问的动作空间无限，直接对文本做 RL 不可行。这里把「提问」抽象为
  「从有限候选库里选一条」，策略只在有限动作集上决策——这是让 RL 落到
  非结构化文本任务上的标准做法（动作抽象 / action abstraction）。

动作库设计：
  由题库 golden_qa 关键词派生，刻意混入「信息型」与「无关型」动作：
    信息型：肉 / 获救（题001）、热气球 / 抽签（题002）→ 主持答「是」
    无关型：钱 / 水                                  → 主持答「无关」

实测可学信号（2026-10-12 · 务必按事实陈述，勿照抄设计预期）：
  ① **不重复已问动作**——稳定可学。5 个随机种子全部成立（见
     tests/test_rl.py）：已问动作的 Q 显著低于未问动作。
  ② **偏好信息型动作**——⚠️ **学不出来（已证伪的负向结论）**。
     设计预期是「信息型平均 +0.2、无关型 −0.2，存在 0.4 差距」，但实测：
     eps∈{0.15,0.3,0.5} × 局数∈{400,800,1200,1500,3000} 共 5 组配置，
     且额外做了「每局随机抽题」以打散题目轮换与动作选择的相位耦合，
     结果 12 组中 8 组失败、动作排序随种子乱跳（无关动作「水」甚至排到过第一）。
     原因：每个信息型动作只在 3 题中的 1 题上有效，单步奖励方差极大
     （+1 或 −0.2），叠加贪心自强化导致各动作访问次数悬殊、Q 估计被带偏；
     终局 F1 奖励还带题目特异性，进一步掩盖那 0.4 的差距。
     → 要让②可学，需要「在多数题目上都有效」的动作（更大题库 / 语义化动作），
       而非当前与单题绑定的关键词动作。

已知局限（诚实）：
  本模块当前价值是打通「L2 策略可训练 + 可评测 + 可对比」的完整链路，
  并给出一条可复现的学习证据（贪心 > 随机，由信号①驱动）。
  **不宣称**策略已具备通用提问或推理能力。
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .base import Player
from engine.game import Puzzle
from engine.evaluate import char_f1

# 动作库：信息型（前 4）+ 无关型（后 2），见模块文档
ACTION_BANK: List[str] = ["肉", "获救", "热气球", "抽签", "钱", "水"]
UNINFORMATIVE_ANSWERS = ("无关", "不重要")

REPEAT_PENALTY = -0.5
UNINFORMATIVE_PENALTY = -0.2
INFORMATIVE_REWARD = 1.0
TERMINAL_F1_WEIGHT = 2.0

POLICY_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "rl_policy.json",
)


def render_question(keyword: str) -> str:
    """把动作（关键词）渲染成自然语言提问。

    渲染结果必须包含关键词本身：MockHost 按 golden_qa 的 contains 做子串
    匹配，关键词不在问句里就永远拿不到有效回答，RL 也就没有学习信号。
    """
    return f"这件事和「{keyword}」有关吗？"


def step_reward(is_repeat: bool, answer: str) -> float:
    """单步奖励：重复提问罚最重，无信息次之，有信息为正。"""
    if is_repeat:
        return REPEAT_PENALTY
    if answer in UNINFORMATIVE_ANSWERS:
        return UNINFORMATIVE_PENALTY
    return INFORMATIVE_REWARD


def terminal_reward(guess: str, solution: str) -> float:
    """终局奖励：还原度（字符级 F1）加权。

    F1 由环境给出、用作奖励；策略本身**不观测** solution，不存在答案泄漏。
    """
    return TERMINAL_F1_WEIGHT * char_f1(guess, solution)


class TabularQPolicy:
    """离散动作空间的表格 Q 学习（epsilon-greedy）。

    state = 已问动作集合（排序元组转字符串键）。选择「集合」而非「序号」
    是为了让「不要重复提问」成为可学习状态特征——若状态不含已问信息，
    策略无法区分重复与新问。
    """

    def __init__(self, n_actions: int, alpha: float = 0.35, gamma: float = 0.9,
                 epsilon: float = 0.15, seed: int = 0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: Dict[str, List[float]] = {}
        self._rng = random.Random(seed)

    def _key(self, state: Sequence[int]) -> str:
        return ",".join(str(i) for i in sorted(set(state)))

    def _q(self, state: Sequence[int]) -> List[float]:
        k = self._key(state)
        if k not in self.Q:
            self.Q[k] = [0.0] * self.n_actions
        return self.Q[k]

    def select_action(self, state: Sequence[int], explore: bool = True) -> int:
        """epsilon-greedy 选动作；explore=False 或 epsilon=0 时为纯贪心。"""
        q = self._q(state)
        if explore and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        # 贪心：并列时取 index 最小者（确定性，便于复现）
        return max(range(self.n_actions), key=lambda a: (q[a], -a))

    def greedy_action(self, state: Sequence[int]) -> int:
        return self.select_action(state, explore=False)

    def update(self, state: Sequence[int], action: int, reward: float,
               next_state: Sequence[int], done: bool = False) -> None:
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
                 explore: bool = True):
        super().__init__(id, model)
        self.bank: List[str] = list(action_bank or ACTION_BANK)
        self.policy = policy or TabularQPolicy(len(self.bank))
        self.explore = explore
        # 训练回路需要知道「这一步用了哪个状态/动作」，由 ask_question 记录
        self.last_state: Tuple[int, ...] = tuple()
        self.last_action: int = -1

    def _asked_actions(self, transcript: Sequence[dict]) -> List[int]:
        """从对局记录反推已问动作（无内部状态依赖，可安全用于任意引擎）。"""
        asked: List[int] = []
        for m in transcript:
            if m.get("role") != "question" or m.get("speaker") != self.id:
                continue
            content = m.get("content", "")
            for i, kw in enumerate(self.bank):
                if f"「{kw}」" in content:
                    asked.append(i)
                    break
        return asked

    def ask_question(self, puzzle: Puzzle, transcript: List[dict]) -> str:
        asked = self._asked_actions(transcript)
        state = tuple(sorted(set(asked)))
        action = self.policy.select_action(state, explore=self.explore)
        self.last_state = state
        self.last_action = action
        return render_question(self.bank[action])

    def final_guess(self, puzzle: Puzzle, transcript: List[dict]) -> str:
        """终局还原：把问答证据串成一段话（确定性，不调用模型）。

        刻意不加入汤面原文——汤面与汤底字符重合度低，塞进去只会拉低
        精确率、污染 F1 信号，让奖励变得不可解释。
        """
        pairs: List[Tuple[str, str]] = []
        pending: Optional[str] = None
        for m in transcript:
            role = m.get("role")
            if role == "question":
                pending = m.get("content", "")
            elif role == "answer" and pending is not None:
                pairs.append((pending, m.get("content", "")))
                pending = None
        body = "；".join(f"{q}→{a}" for q, a in pairs)
        return f"【RL 策略还原】根据问答：{body}。我推断真相与上述线索直接相关。"


def run_episode(puzzle: Puzzle, host, questioner: RLQuestioner,
                max_rounds: int = 4, train: bool = True) -> dict:
    """跑一局（RL 回路），返回 {reward, f1, transcript}。

    train=True 时做 Q 更新。训练与评估共用本函数，避免两套循环实现漂移。
    提问走 questioner.ask_question（而非直接调 policy），确保 Player 接口
    与真实引擎 TurtleSoupGame 的行为完全一致。
    """
    transcript: List[dict] = []
    asked: List[int] = []
    total = 0.0
    traj: List[Tuple[Tuple[int, ...], int, float]] = []

    for r in range(1, max_rounds + 1):
        question = questioner.ask_question(puzzle, transcript)
        action = questioner.last_action
        state = questioner.last_state
        transcript.append({"round": r, "speaker": questioner.id,
                           "role": "question", "content": question})

        answer = host.answer(puzzle, question, transcript)
        transcript.append({"round": r, "speaker": getattr(host, "id", "host"),
                           "role": "answer", "content": answer})

        is_repeat = action in asked
        rew = step_reward(is_repeat, answer)
        asked.append(action)
        traj.append((state, action, rew))
        total += rew

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
            questioner.policy.update(state, action, G, tuple(), done=True)

    return {"reward": total, "f1": f1, "transcript": transcript, "guess": guess}
