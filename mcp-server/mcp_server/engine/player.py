"""玩家适配层：统一接口调多种模型（litellm），并以离线玩家兜底。

三类玩家的分工（v0.2）：

- **MockPlayer**：知道真相的"预言机"，仅用于验证端到端流程，**不用于能力评测**。
- **HeuristicPlayer**：不作弊的启发式玩家。只拿到汤面(surface)，靠「汤面词汇 +
  常识假设」提问，用裁判反馈(yes/no)累积线索并重构猜想。因不同性格策略的提问
  数量与角度不同，得分会产生**真实区分度**——这才让排行榜有意义。
- **LLMPlayer**：通过 litellm 调真实模型，解析其 JSON 动作；失败时安全降级。

工厂：get_player(model, puzzle) —— 按名字前缀分发。
"""
from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from mcp_server.config import has_api_key, resolve_model
from mcp_server.engine.judge import _STOP
from mcp_server.engine.prompts import build_player_prompt

logger = logging.getLogger("ai-arena.player")

try:
    import litellm  # 可选依赖
    _LITELLM_AVAILABLE = True
except Exception:  # noqa: BLE001
    _LITELLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# 启发式玩家的「常识假设库」：海龟汤高频真相方向
# 汤面往往不直接包含真相词汇，纯靠汤面分词提问命中率很低；
# 补充这些通用维度后，玩家才可能触及真相事实（这也是真实玩家的推理方式）。
# ---------------------------------------------------------------------------
_GENERIC_HYPOTHESES: list[str] = [
    "人物身份被隐瞒",
    "职业是关键线索",
    "存在一场误会",
    "时间发生了错位",
    "有人在伪装",
    "涉及死亡或失踪",
    "地点另有隐情",
    "日常物品另有用途",
    "现场有第二个人",
    "记忆出了问题",
    "身份被替换",
    "源于某个约定",
    "与亲情有关",
    "有人说谎",
    "可以用科学原理解释",
    "是心理作用导致",
]


# ---------------------------------------------------------------------------
# 性格策略：决定提问规模、通用假设占比、何时收网猜测
# ---------------------------------------------------------------------------
_STYLES: dict[str, dict[str, Any]] = {
    # 均衡：12 个词、三成通用假设、交替提问、六成进度收网、用 5 条线索作答
    "balanced": {"term_limit": 12, "generic_ratio": 0.30, "guess_after": 0.60,
                 "order": "interleave", "max_guess_terms": 5, "repeat_rate": 0.15},
    # 激进：先赌大方向、问得少、收网早、只用 2 条线索就敢下结论、几乎不回头复核
    "aggressive": {"term_limit": 7, "generic_ratio": 0.20, "guess_after": 0.30,
                   "order": "generic_first", "max_guess_terms": 2, "repeat_rate": 0.05},
    # 保守：先穷举汤面细节、问得多、收网晚、线索全部用上、**最爱回头复核**
    "conservative": {"term_limit": 18, "generic_ratio": 0.40, "guess_after": 0.85,
                     "order": "term_first", "max_guess_terms": None, "repeat_rate": 0.30},
    # 跳跃：大量通用假设、顺序打乱、线索全部用上 —— 覆盖广、方差大
    "creative": {"term_limit": 10, "generic_ratio": 0.65, "guess_after": 0.55,
                 "order": "shuffle", "max_guess_terms": None, "repeat_rate": 0.10},
}


class BasePlayer:
    def __init__(self, name: str, puzzle: dict[str, Any]):
        self.name = name
        self.puzzle = puzzle
        # 是否偷看真相（用于评测可信度自检；作弊玩家不得参与排名）
        self.uses_truth = False

    def act(self, state: dict[str, Any]) -> dict[str, str]:
        raise NotImplementedError


class MockPlayer(BasePlayer):
    """预言机玩家：依次就每个关键事实提问（让汤主揭示），最后用已知事实完美重构猜想。

    仅用于验证端到端流程，因其"作弊"式掌握真相，不应用于能力评测。
    """

    def __init__(self, name: str, puzzle: dict[str, Any]):
        super().__init__(name, puzzle)
        self.uses_truth = True
        self.facts: list[str] = puzzle.get("key_facts", [])
        self._i = 0

    def act(self, state: dict[str, Any]) -> dict[str, str]:
        if self._i < len(self.facts):
            fact = self.facts[self._i]
            self._i += 1
            return {"action": "ask", "content": f"这件事是不是和「{fact}」有关？"}
        # 已问完所有事实，提交完美重构
        return {"action": "guess", "content": "。".join(self.facts)}


class HeuristicPlayer(BasePlayer):
    """启发式玩家：不偷看真相，仅依据汤面与裁判反馈推理。

    行为流程：
    1. 从汤面抽取候选词，结合常识假设库生成提问队列（问题刻意保持简短，
       以提高对事实的覆盖率命中率）。
    2. 每轮提一个问题，把裁判回答为 yes 的线索记入"已确认线索"。
    3. 进度达到 guess_after 后收网，用累积线索重构猜想。

    因性格策略不同（提问数量/通用假设占比/收网时机），最终命中率天然分化，
    无需 API key 即可产生有区分度的排行榜。
    """

    def __init__(self, name: str, puzzle: dict[str, Any]):
        super().__init__(name, puzzle)
        self.uses_truth = False
        # 关键：只保留汤面，绝不持有 truth / key_facts
        self.surface: str = puzzle.get("surface", "")

        # 风格名分隔符同时支持连字符与冒号（heuristic-aggressive / heuristic:aggressive），
        # 未识别时退回 balanced——绝不静默变成另一种策略。
        tail = name.lower().split("heuristic", 1)[1].lstrip(":-_ ")
        self.style_key = tail if tail in _STYLES else "balanced"
        style = _STYLES[self.style_key]

        self.term_limit: int = style["term_limit"]
        self.generic_ratio: float = style["generic_ratio"]
        self.guess_after: float = style["guess_after"]
        # 收网时使用多少条线索：None 表示全部用上，数字表示只用前 N 条（越草率）
        self.max_guess_terms: int | None = style["max_guess_terms"]

        # 问题 -> 该问题对应的线索词（便于从 yes 回推线索，避免模板词污染猜想）
        self._q_term: dict[str, str] = {}
        self._queue: list[str] = self._build_queue(style["order"])
        self._cur = 0
        self._seen: set[str] = set()
        self._asked = 0
        self._confirmed: list[str] = []
        # deception 模式测谎用：问题 -> 最近一次回答；以及发现的"前后不一致"问题队列
        self._qa: dict[str, str] = {}
        self._conflicts: list[str] = []
        self._challenged: set[str] = set()
        # 回头复核（重复提问）的概率：越高越容易识破谎言
        self.repeat_rate: float = style["repeat_rate"]

    # -- 提问队列构建 ------------------------------------------------------
    def _build_queue(self, order: str) -> list[str]:
        """按性格决定提问顺序——这是区分度的关键来源。

        若各性格只是"提问数量"不同而顺序一致，先猜的人与后猜的人拿到的线索集合
        会是子集关系，得分极易打平。让顺序也分化（先赌方向 / 先抠细节 / 打乱），
        不同时刻累积的线索才真正不同。
        """
        terms = _extract_terms(self.surface, limit=self.term_limit)
        term_qs: list[str] = []
        for t in terms:
            q = _make_question(t)
            self._q_term[q] = t  # 记录问题->线索词，供回推确认线索
            term_qs.append(q)

        generic = list(_GENERIC_HYPOTHESES)
        if order == "shuffle":
            random.shuffle(generic)
        n_generic = max(1, int(len(term_qs) * self.generic_ratio))
        generic_qs: list[str] = []
        for g in generic[:n_generic]:
            q = _make_question(g)
            self._q_term[q] = g
            generic_qs.append(q)

        if order == "shuffle":
            out = term_qs + generic_qs
            random.shuffle(out)
            return out
        if order == "generic_first":
            return generic_qs + term_qs
        if order == "term_first":
            return term_qs + generic_qs

        # interleave：以具体词为主，均匀穿插通用假设
        if not generic_qs:
            return term_qs
        step = max(1, len(term_qs) // len(generic_qs))
        out: list[str] = []
        gi = 0
        for i, q in enumerate(term_qs):
            out.append(q)
            if (i + 1) % step == 0 and gi < len(generic_qs):
                out.append(generic_qs[gi])
                gi += 1
        out.extend(generic_qs[gi:])
        return out

    # -- 行动 --------------------------------------------------------------
    def act(self, state: dict[str, Any]) -> dict[str, str]:
        self._sync_confirmed(state)
        self._sync_qa(state)

        # deception 模式：先检查是否抓到了前后矛盾——那就是谎言的证据
        if state.get("mode") == "deception":
            conflict = self._detect_conflict()
            if conflict:
                term = self._q_term.get(conflict, conflict)
                return {
                    "action": "challenge",
                    "content": f"关于「{term}」你前后回答不一致，你在撒谎。",
                    "target": conflict,  # 供引擎核验：指出到底哪个问题前后矛盾
                }

        round_ = state.get("round", 1)
        max_rounds = max(int(state.get("max_rounds", 20)), 1)
        progress = (round_ - 1) / max_rounds

        mode = state.get("mode")
        # debate 模式下收网提交的是"论证"，不是单纯的猜测
        closing = "argue" if mode == "debate" else "guess"

        # 收网条件：进度达标且至少问过少量问题；或问题已问尽
        if (progress >= self.guess_after and self._asked >= 3) or self._asked >= len(self._queue):
            return {"action": closing, "content": self._build_guess()}

        q = self._next_question(mode)
        if q is None:
            return {"action": closing, "content": self._build_guess()}
        self._asked += 1
        return {"action": "ask", "content": q}

    def _next_question(self, mode: str | None = None) -> str | None:
        # deception 模式下按概率"回头复核"：把问过的问题再问一遍。
        # 若这次的回答与上次不同，就抓到了谎言——这是真人玩海龟汤时的常用手法。
        if mode == "deception" and self._seen and random.random() < self.repeat_rate:
            return random.choice(sorted(self._seen))

        while self._cur < len(self._queue):
            q = self._queue[self._cur]
            self._cur += 1
            if q not in self._seen:
                self._seen.add(q)
                return q
        # 队列问尽：deception 模式下转为反复复核，继续找矛盾
        if self._seen:
            return random.choice(sorted(self._seen))
        return None

    # -- 线索与猜想 --------------------------------------------------------
    def _sync_confirmed(self, state: dict[str, Any]) -> None:
        """把裁判判为 yes 的提问对应的线索词记入已确认线索。"""
        for rec in state.get("transcript", []):
            if rec.get("player") != self.name:
                continue
            if rec.get("host_answer") != "yes":
                continue
            term = self._q_term.get(rec.get("content", ""), rec.get("content", ""))
            if term and term not in self._confirmed:
                self._confirmed.append(term)

    def _sync_qa(self, state: dict[str, Any]) -> None:
        """记录自己每个提问得到的回答，发现同一问题前后答案不同即记为矛盾。"""
        for rec in state.get("transcript", []):
            if rec.get("player") != self.name or rec.get("action") != "ask":
                continue
            ans = rec.get("host_answer")
            q = rec.get("content", "")
            if not ans or not q:
                continue
            if q in self._qa and self._qa[q] != ans and q not in self._challenged:
                self._conflicts.append(q)
            self._qa[q] = ans

    def _detect_conflict(self) -> str | None:
        """取出一个尚未质疑过的矛盾问题（返回问题原文，供引擎核验）。"""
        while self._conflicts:
            q = self._conflicts.pop(0)
            if q in self._challenged:
                continue
            self._challenged.add(q)
            return q
        return None

    def _build_guess(self) -> str:
        """收网：用已确认线索重构猜想。

        性格在这里第三次分化——激进派只拿前 2 条线索就敢下结论（草率），
        保守派要把线索用全（扎实）。即使提问集合相同，最终得分也会不同。
        """
        if not self._confirmed:
            return "线索不足，我猜测这件事的核心是一场误会。"
        terms = self._confirmed
        if self.max_guess_terms is not None:
            terms = terms[: self.max_guess_terms]
        body = "；".join(terms)
        return f"综合已确认的线索：{body}。真相应由这些线索共同构成。"

    def remember_question_term(self, question: str, term: str) -> None:
        self._q_term[question] = term


# 知情者策略（deception 模式）：撒谎比例
_INSIDER_STRATEGIES: dict[str, dict[str, float]] = {
    "honest": {"lie_rate": 0.0},      # 始终如实 —— 对照组
    "misleading": {"lie_rate": 1.0},  # 始终反向 —— 激进欺骗，但规律明显易被识破
    "mixed": {"lie_rate": 0.5},       # 真话假话混着说 —— 最难对付
}


class InsiderPlayer(BasePlayer):
    """知情者（deception 模式专用）：掌握真相，负责回答求解者的提问。

    与 MockPlayer 的本质区别：
    - MockPlayer 是**作弊的求解者**（本不该知道答案却知道），会污染评测；
    - InsiderPlayer 的知情是**角色设定**——正如真人海龟汤里的"汤主"本就知道答案，
      他的任务不是答题而是误导。因此它不参与求解者排名，只评估欺骗/伪装能力。

    策略：honest（如实，做对照）/ misleading（反向，激进）/ mixed（摇摆，最难）。
    """

    def __init__(self, name: str, puzzle: dict[str, Any]):
        super().__init__(name, puzzle)
        self.uses_truth = True  # 角色设定使然，不是作弊
        self.role = "insider"

        key = name.lower().split("insider", 1)[1].lstrip("-") or "mixed"
        self.strategy_key = key if key in _INSIDER_STRATEGIES else "mixed"
        self.lie_rate = _INSIDER_STRATEGIES[self.strategy_key]["lie_rate"]
        self.answer_count = 0
        self.lie_count = 0

    def answer(self, question: str, truth_answer: str) -> tuple[str, bool]:
        """回答一个提问。truth_answer 是裁判给出的真实答案（ground truth）。

        返回 (对外给出的回答, 是否撒谎)。撒谎即对真实答案取反。
        """
        self.answer_count += 1
        lie = random.random() < self.lie_rate
        if lie:
            self.lie_count += 1
            return _flip_answer(truth_answer), True
        return truth_answer, False

    def act(self, state: dict[str, Any]) -> dict[str, str]:
        # 知情者不主动行动：回合只在求解者之间轮转，他只负责应答
        return {"action": "pass", "content": ""}


class LLMPlayer(BasePlayer):
    """真实模型玩家：用 litellm 统一接口，解析 JSON 动作。"""

    def __init__(self, name: str, puzzle: dict[str, Any]):
        super().__init__(name, puzzle)
        self.uses_truth = False
        if not _LITELLM_AVAILABLE:
            raise RuntimeError("litellm 未安装，无法使用 LLMPlayer（pip install 'ai-arena[llm]'）")
        self.model_str, self._env = resolve_model(name)
        if not has_api_key(name):
            raise RuntimeError(f"模型 {name} 未配置 API key（检查 .env 的 {self._env}）")

    def act(self, state: dict[str, Any]) -> dict[str, str]:
        prompt = build_player_prompt(state, self.name)
        try:
            resp = litellm.completion(
                model=self.model_str,
                messages=[
                    {"role": "system", "content": "你是海龟汤玩家，严格按 JSON 格式输出。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            action = data.get("action")
            text = data.get("content", "")
            if action in ("ask", "guess") and text:
                return {"action": action, "content": text}
        except Exception as e:  # noqa: BLE001
            logger.warning("LLMPlayer %s 调用失败，降级为保守提问: %s", self.name, e)
        # 安全降级：返回一次保守提问，保证对局不中断
        return {"action": "ask", "content": "你能再给一点线索吗？"}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _extract_terms(surface: str, limit: int = 12) -> list[str]:
    """从汤面抽取候选线索词：按 2 字**不重叠**切分。

    为什么不用滑动窗口：逐字滑动会产生「男人每 / 人每天」这类跨词边界的噪声片段，
    提问质量很差。按 2 字对齐切分更贴近中文词边界（男人 / 每天 / 深夜 / 地铁），
    再过滤含停用字者，得到的候选词可读性明显更好。
    """
    text = re.sub(r"[^\u4e00-\u9fa5]", "", surface)
    seen: set[str] = set()
    terms: list[str] = []
    size = 2
    for i in range(0, len(text) - size + 1, size):
        t = text[i : i + size]
        if t in seen or any(ch in _STOP for ch in t):
            continue
        seen.add(t)
        terms.append(t)
    return terms[:limit]


def _make_question(term: str) -> str:
    """生成简短提问：问题越短，对事实的覆盖率越高，越容易命中。"""
    return f"{term}是关键吗？"


def _flip_answer(ans: str) -> str:
    """把 yes/no 取反（知情者撒谎用）。"""
    return "no" if ans == "yes" else "yes"


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def get_player(model: str, puzzle: dict[str, Any]) -> BasePlayer:
    """按名字前缀分发玩家实现。

    - heuristic* → HeuristicPlayer（离线、不作弊，用于有区分度的评测）
    - mock* / 无 key → MockPlayer（离线跑通流程）
    - 其他 → LLMPlayer（真实模型，失败自动降级）
    """
    name = model.lower()
    if name.startswith("insider"):
        return InsiderPlayer(model, puzzle)
    if name.startswith("heuristic"):
        return HeuristicPlayer(model, puzzle)
    if name.startswith("mock") or not has_api_key(model):
        return MockPlayer(model, puzzle)
    try:
        return LLMPlayer(model, puzzle)
    except Exception as e:  # noqa: BLE001
        logger.warning("无法创建 LLMPlayer(%s)，改用 MockPlayer: %s", model, e)
        return MockPlayer(model, puzzle)
