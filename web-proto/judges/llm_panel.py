"""真实 LLM 裁判面板（P2-A / A-7）——回答「受控结论对真实裁判是否成立」。

## 三条设计要点（改代码前先读 DESIGN-P2A7-REAL-JUDGE.md）

1. **异质性来自裁判人格 prompt，不是不同模型**：本机只有一个可用端点，
   故 `strict / lenient / keyword / holistic` 四人格同源同模型。
   ⇒ 本层结论只覆盖「同源异质」，**不可**外推为「跨模型裁判共识的增益」。

2. **调用失败 → 缺席裁判，绝不折算成 0 分**：既有 `players/llm.py::LLMJudge`
   在失败时 `return 0.0`，那会让「调用失败」冒充「裁判打了 0 分」，
   系统性压低聚合值——本项目反复栽过的「静默 fallback 比报错危险」。
   本模块抛 `JudgeUnavailable`，由 `RealJudgePanel` 映射为 P2-A 既有的
   `absent` 机制（响应权重比 < 2/3 → **拒绝发布**），即失败会真实影响结论。

3. **结果按 (solution, guess) 缓存**：标定与共识聚合会两次访问同一用例，
   `temperature=0` 下结果确定，缓存保证不重复烧 API 费用；
   **失败也进缓存**，避免第二次调用把「第一次失败」变成「第二次成功」。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 允许从任意工作目录 import（Windows 路径分隔符也要覆盖，故不用 rsplit("/")）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.evaluate import char_f1  # noqa: E402
from players.base import Judge  # noqa: E402
from players.llm import _call, LLMError  # noqa: E402

from .panel import JudgePanel, SignedVerdict  # noqa: E402


class JudgeUnavailable(RuntimeError):
    """裁判不可用（调用失败 / 返回不可解析 / 缺分数）。**不是分数。**"""


# --------------------------------------------------------------------------
# 裁判人格（同模型、不同打分倾向）
# --------------------------------------------------------------------------

PERSONAS: Dict[str, str] = {
    "strict": (
        "你是**严格**的海龟汤还原度裁判。只按「汤底中被明确还原的关键事实比例」"
        "打分：遗漏关键事实要扣分，含糊表述按未还原计。宁可给低分，不要给人情分。"
    ),
    "lenient": (
        "你是**宽松**的海龟汤裁判。只要还原方向大体正确、即使细节有偏差，"
        "也给较高分；关注整体 plausibility，不苛求逐字对应。"
    ),
    "keyword": (
        "你是**字面命中**裁判。忽略整体通顺与否，只数玩家还原中出现了多少"
        "汤底里的关键实体、动作、因果词；命中越多分越高，不命中即 0 贡献。"
    ),
    "holistic": (
        "你是**叙事完整性**裁判。只看玩家还原是否讲出一个自洽、有因果的"
        "故事；个别细节缺失不扣分，但逻辑矛盾或明显跑题要大幅扣分。"
    ),
}


def _clip_score(v: Any) -> Optional[float]:
    """把裁判返回值收敛到 [0,100]；非数或越界到无法解释时判为不可用。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return max(0.0, min(100.0, f))


# thinking 模型（glm-4.5 等）会把 max_tokens 全消耗在 reasoning 上，
# 实测 finish_reason=length / content 为空 / reasoning_content 1399 字符
# ——调用方看到的是 empty_content，必须显式关思考才能拿到 JSON 正文。
DISABLE_THINKING = {"thinking": {"type": "disabled"}}


def call_judge(model: str, system: str, conversation: List[dict], *,
               temperature: float, max_tokens: int, retries: int = 2) -> str:
    """裁判专用调用：端点支持时关思考，不支持则放大预算重试一次。"""
    extra = DISABLE_THINKING if model.startswith(("zai/", "zhipuai/")) else None
    try:
        return _call(model, system, conversation, temperature=temperature,
                     max_tokens=max_tokens, retries=retries,
                     extra_body=extra)
    except LLMError:
        if extra is None:
            raise
        # 端点不认 extra_body（或思考无法关闭）：给足预算再来一次
        return _call(model, system, conversation, temperature=temperature,
                     max_tokens=max_tokens * 4, retries=1)


class LLMJudgeNode(Judge):
    """一个真实 LLM 裁判（固定人格、temperature=0、结果缓存）。"""

    def __init__(self, judge_id: str, model: str,
                 persona: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 256,
                 retries: int = 2):
        if judge_id not in PERSONAS:
            raise ValueError(f"未知裁判人格：{judge_id}（可选 {list(PERSONAS)}）")
        self.id = judge_id
        self.model = model
        self.persona = persona or PERSONAS[judge_id]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self._cache: Dict[Tuple[str, str], Any] = {}
        # 诊断计数（报告用）
        self.n_calls = 0
        self.n_unavailable = 0

    # -- Judge 契约 ----------------------------------------------------
    def score(self, puzzle, guess: str) -> Tuple[float, str]:
        key = (str(getattr(puzzle, "solution", "")), str(guess))
        if key in self._cache:
            cached = self._cache[key]
            if isinstance(cached, JudgeUnavailable):
                raise cached
            return cached

        system = (
            f"{self.persona}\n\n"
            f"真实汤底：{getattr(puzzle, 'solution', '')}\n"
            f"玩家还原：{guess}\n\n"
            "给出 0-100 的分数（可带小数）与一句简短评语。"
            "只输出 JSON：{\"score\": <number>, \"rationale\": \"<str>\"}"
        )
        try:
            raw = call_judge(self.model, "", [{"role": "user", "content": system}],
                             temperature=self.temperature,
                             max_tokens=self.max_tokens, retries=self.retries)
        except (LLMError, Exception) as exc:  # 网络/鉴权/超时一律「不可用」
            self.n_unavailable += 1
            err = JudgeUnavailable(
                f"{self.id} 调用失败 {type(exc).__name__}: {str(exc)[:120]}")
            self._cache[key] = err
            raise err from exc

        try:
            start, end = raw.find("{"), raw.rfind("}") + 1
            obj = json.loads(raw[start:end])
        except Exception as exc:
            self.n_unavailable += 1
            err = JudgeUnavailable(f"{self.id} 返回不可解析：{raw[:100]!r}")
            self._cache[key] = err
            raise err from exc

        score = _clip_score(obj.get("score"))
        if score is None:
            self.n_unavailable += 1
            err = JudgeUnavailable(f"{self.id} 分数字段不可用：{obj!r}")
            self._cache[key] = err
            raise err          # 注意：此处无活跃的 except 块，不能 `from exc`

        out = (round(score, 4), str(obj.get("rationale", ""))[:200])
        self._cache[key] = out
        self.n_calls += 1
        return out

    def reset_rng(self) -> None:
        """temperature=0 下无 RNG 可复位；保留接口是为了让标定可重跑。

        刻意**不清缓存**：清掉会让「同用例重复计费」重新发生。
        重跑应构造新节点。
        """

    @property
    def n_cache(self) -> int:
        return len(self._cache)


class RealJudgePanel(JudgePanel):
    """真实 LLM 裁判面板：失败映射为缺席，聚合/共识复用 P2-A 内核。"""

    def __init__(self, judges: Sequence[Any], model: str,
                 chain_path: Optional[str] = None, sign: bool = False,
                 temperature: float = 0.0):
        # 只把「字符串 id」当作人格构造节点；已是裁判对象的原样保留
        # （测试用假裁判、未来的外部 policy 都走这条路）
        nodes = [LLMJudgeNode(j, model, temperature=temperature)
                 if isinstance(j, str) else j
                 for j in judges]
        super().__init__(nodes, sign=sign, chain_path=chain_path)
        self.last_absent: List[str] = []
        self.unusable: List[str] = []   # 标定覆盖率不足的裁判

    # -- 评分：逐个捕获失败 -------------------------------------------
    def score_all(self, puzzle, guess: str,
                  absent: Sequence[str] = ()) -> List[SignedVerdict]:
        absent_set = set(absent)
        out: List[SignedVerdict] = []
        failed: List[str] = []
        for j in self.judges:
            if j.id in absent_set or j.id in self.unusable:
                failed.append(j.id)
                continue
            try:
                sc, rationale = j.score(puzzle, guess)
            except JudgeUnavailable:
                failed.append(j.id)
                continue
            if self.sign:
                s = self._signers[j.id]
                sig = s.sign_hex(json.dumps(
                    {"judge_id": j.id, "score": sc},
                    ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode("utf-8"))
                out.append(SignedVerdict(
                    judge_id=j.id, score=sc, rationale=rationale,
                    signature_hex=sig, public_key_hex=s.public_key_hex,
                    algorithm=s.algorithm, key_custody=s.key_custody))
            else:
                out.append(SignedVerdict(
                    judge_id=j.id, score=sc, rationale=rationale,
                    signature_hex="", public_key_hex="",
                    algorithm="unsigned", key_custody="n/a"))
        self.last_absent = sorted(set(failed))
        return out

    # -- 标定：覆盖率不足的裁判判为不可用 ------------------------------
    def calibrate_safe(self, labeled: Sequence[Tuple[Any, str]],
                      min_coverage: float = 0.5) -> Dict[str, float]:
        """标定并返回 {judge_id: rmse}；覆盖率不足的裁判进 `unusable`。

        与父类 `calibrate` 的差别：**不会**因单个裁判的一次失败而整体崩掉，
        也不会把失败当 0 分计入（那会系统性低估 RMSE）。
        """
        if not labeled:
            raise ValueError("标定集不能为空（无 ground truth 则无法标定）")
        for j in self.judges:
            j.reset_rng()
            j.err2 = []  # type: ignore[attr-defined]

        for puzzle, guess in labeled:
            truth = char_f1(guess, puzzle.solution) * 100.0
            for j in self.judges:
                try:
                    sc, _ = j.score(puzzle, guess)
                except JudgeUnavailable:
                    continue  # 跳过该样本，不折算成分数
                j.err2.append((sc - truth) ** 2)  # type: ignore[attr-defined]

        n_total = len(labeled)
        rmse: Dict[str, float] = {}
        self.unusable = []
        coverage: Dict[str, float] = {}
        for j in self.judges:
            errs = getattr(j, "err2", [])
            coverage[j.id] = len(errs) / n_total
            if not errs or coverage[j.id] < min_coverage:
                self.unusable.append(j.id)
                continue
            rmse[j.id] = (sum(errs) / len(errs)) ** 0.5
        if not rmse:
            raise ValueError("所有裁判覆盖率均不足，无法标定（端点可能已不可用）")
        self.calibration = dict(rmse)
        self._coverage = coverage
        return dict(rmse)

    def coverage(self, labeled: Optional[Sequence[Tuple[Any, str]]] = None
                 ) -> Dict[str, float]:
        """各裁判在标定集上的可用覆盖率（报告用）。

        标定后直接取标定时算出的值（唯一可信口径）；未标定时若给了用例集
        则现算，否则返回空字典——**不猜、不用缓存条数冒充覆盖率**。
        """
        cov = getattr(self, "_coverage", None)
        if cov is not None:
            return dict(cov)
        if labeled is None:
            return {}
        n_total = max(1, len(labeled))
        keys = {(str(getattr(p, "solution", "")), str(g))
                for p, g in labeled}
        out: Dict[str, float] = {}
        for j in self.judges:
            hit = sum(1 for k in keys
                      if k in j._cache and not isinstance(j._cache[k], JudgeUnavailable))
            out[j.id] = hit / n_total
        return out

    # -- 共识：先探测缺席，再走父类内核 --------------------------------
    def verdict(self, puzzle, guess: str, method: str = "weighted",
                absent: Sequence[str] = ()):
        """跑一次共识。调用两次 score_all，但第二次命中缓存（无额外 API 调用）。"""
        self.score_all(puzzle, guess, absent=absent)      # 探测缺席集合
        return self.consensus_verdict(puzzle, guess,
                                      absent=tuple(self.last_absent),
                                      method=method)


def make_real_panel(persona_ids: Optional[Sequence[str]] = None,
                    model: str = "zai/glm-4.5-flash",
                    sign: bool = False,
                    temperature: float = 0.0) -> RealJudgePanel:
    ids = list(persona_ids or PERSONAS.keys())
    return RealJudgePanel(
        [LLMJudgeNode(pid, model, temperature=temperature) for pid in ids],
        model=model, sign=sign, temperature=temperature)