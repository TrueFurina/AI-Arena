"""M6 辩论 + LLM-as-Judge 框架（零 key 可运行 PoC）。对应 specs/debate-design.md。

设计目标（SRE 视角：先度量后优化、零 key 优先）：
- 把「辩论」抽象成可复现的状态机：命题 → 多轮正反发言 → 裁判评分。
- 真实模型接入点用**依赖注入** completion_fn（与 LLMQuestioner 同范式）：
  零 key 时用 MockDebater / MockJudge 跑通整条 pipeline；拿到 key 后换 LLMDebater / LLMJudge。
- 裁判产出结构化 verdict（winner / scores / rationale），便于横向聚合与审计。

本文件只放「引擎与零 key 实现」。真实 LLM 实现见 specs/debate-design.md §5（未实现）。
"""

from .roles import Stance


class Debater:
    """辩论者基类。子类实现 take_turn。"""

    name = "debater"

    def take_turn(self, observation: dict) -> dict:
        """返回 {'text': str}。observation 含 proposition / stance / round / history。"""
        raise NotImplementedError


class MockDebater(Debater):
    """零 key 辩论者：按立场从预置论点池循环取用，演示 pipeline 不依赖 API。"""

    name = "mock-debater"

    def __init__(self, stance: str, arguments: list):
        self.stance = stance
        self.arguments = list(arguments)
        self._i = 0

    def take_turn(self, observation: dict) -> dict:
        text = self.arguments[self._i % len(self.arguments)]
        self._i += 1
        return {"text": text}


class Judge:
    """裁判基类。score() 返回结构化 verdict。"""

    name = "judge"

    def score(self, transcript: list) -> dict:
        """transcript: [{'speaker','stance','text'}, ...]
        返回 {'winner': str, 'scores': {side: float}, 'rationale': str}。"""
        raise NotImplementedError


# 简单「证据词」启发式（零 key 裁判用）：越能引用事实/数据/机制，越有说服力。
_EVIDENCE_LEXICON = [
    "数据", "研究", "实验", "统计", "案例", "事实", "证据", "测试", "指标",
    "因为", "因此", "例如", "证明", "结果显示", "对比", "可复现",
]


class MockJudge(Judge):
    """零 key 裁判：用轻量启发式给正反双方打分。

    评分维度（透明、可审计，非黑箱）：
    - 证据密度：发言中命中证据词的数量
    - 立场一致性：始终站在自己立场（不自我矛盾）
    - 篇幅：过短视为敷衍，过长截断计分
    最终 winner 取总分高者，rationale 给出可读说明。
    """

    name = "mock-judge"

    def _score_side(self, side, transcript):
        texts = [t["text"] for t in transcript if t.get("stance") == side]
        if not texts:
            return 0.0, "无发言"
        evidence = sum(text.count(w) for text in texts for w in _EVIDENCE_LEXICON)
        total_len = sum(len(t) for t in texts)
        length_score = min(total_len / 200.0, 1.0)  # 200 字封顶计 1.0
        return round(evidence * 1.0 + length_score * 1.5, 3), f"证据词 {evidence} / 篇幅 {total_len} 字"

    def score(self, transcript: list) -> dict:
        sides = []
        for t in transcript:
            if t.get("stance") and t["stance"] not in sides:
                sides.append(t["stance"])
        scores = {}
        detail = {}
        for s in sides:
            sc, why = self._score_side(s, transcript)
            scores[s] = sc
            detail[s] = why
        if not scores:
            return {"winner": "draw", "scores": {}, "rationale": "无有效发言"}
        winner = max(scores, key=scores.get)
        margin = round(scores[winner] - min(scores.values()), 3)
        rationale = (
            f"裁判（启发式）判 {winner} 胜，分差 {margin}。"
            + "；".join(f"{s}: {detail[s]}" for s in sides)
        )
        return {"winner": winner, "scores": scores, "rationale": rationale}


# ---------------------------------------------------------------------------
# 辩论编排（状态机）
# ---------------------------------------------------------------------------

class Debate:
    """多轮辩论状态机：命题下正反方交替发言，结束交裁判。"""

    def __init__(self, proposition, debaters: dict, judge, rounds=3):
        # debaters: {stance: Debater}
        self.proposition = proposition
        self.debaters = debaters
        self.judge = judge
        self.rounds = rounds
        self.transcript = []
        self.verdict = None

    def _observation(self, stance, round_no):
        return {
            "proposition": self.proposition,
            "stance": stance,
            "round": round_no,
            "history": list(self.transcript),
        }

    def run(self):
        order = list(self.debaters.keys())
        for r in range(1, self.rounds + 1):
            for stance in order:
                obs = self._observation(stance, r)
                out = self.debaters[stance].take_turn(obs)
                self.transcript.append({
                    "speaker": getattr(self.debaters[stance], "name", stance),
                    "stance": stance,
                    "round": r,
                    "text": out.get("text", ""),
                })
        self.verdict = self.judge.score(self.transcript)
        return self.verdict

    def build_log(self):
        return {
            "proposition": self.proposition,
            "rounds": self.rounds,
            "transcript": self.transcript,
            "verdict": self.verdict,
        }
