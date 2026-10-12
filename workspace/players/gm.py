"""更聪明的规则化 GM（M2，零 key）。对应 specs/engine-design.md §4/§5.3。

与 MockGM 的区别：
- MockGM 只在「提问逐字包含某关键事实原文」时才答 yes —— 真实模型自由改写就判不出。
- SmartGM 用每题预置的 keywords / clue_keywords 做**关键词重叠**匹配，
  能容错改写（"他打嗝了吗" → 命中事实"男人打嗝"的关键词"打嗝" → yes）。
- 这样还原度才反映「提问者是否探测到了关键事实」，而非「措辞是否逐字复制」。

仍全程规则化、零 key；GM 从不泄漏汤底，只给是非类限定回答。
"""

from .base import GM


class SmartGM(GM):
    name = "smart"

    def answer(self, question, context):
        puzzle = context.get("puzzle") or {}
        key_facts = context.get("key_facts", [])
        clue_gradient = context.get("clue_gradient", [])
        kw = puzzle.get("keywords", {}) if isinstance(puzzle, dict) else {}
        ckw = puzzle.get("clue_keywords", {}) if isinstance(puzzle, dict) else {}

        # 命中关键事实 → yes（回退：无关键词时用事实原文本身）
        for fact in key_facts:
            kws = kw.get(fact, [fact])
            if any(k in question for k in kws):
                return {"answer": "yes"}

        # 命中梯度线索 → yes_and_no（部分相关）
        for clue in clue_gradient:
            kws = ckw.get(clue, [clue])
            if any(k in question for k in kws):
                return {"answer": "yes_and_no"}

        return {"answer": "irrelevant"}


class DeceiverGM(SmartGM):
    """M5 骗子变体：在 SmartGM 之上按比例撒谎（仅翻转 yes/no）。

    设计意图（见 specs/deceiver-design.md）：
    - 让 GM 偶尔给出与真相相反的是非回答，测试提问者是否具备
      「交叉验证 / 矛盾检测」能力，而非盲目相信每一个 yes/no。
    - 撒谎仅在 yes/no 上发生；irrelevant / yes_and_no / unknown 保持原样
      （对「无关问题」撒谎没有信息量，会退化成纯噪声）。
    - 每轮通过返回 dict 的 `lied` 字段如实记录，便于后续度量造假比例，
      但**绝不**把 `lied` 透露给提问者（观察里不含此字段）。

    零 key：纯规则，不需要任何 API。
    """

    name = "deceiver"

    def __init__(self, lie_rate=0.3, rng=None):
        # lie_rate ∈ [0,1]：每轮以该概率翻转 yes/no 答案
        self.lie_rate = max(0.0, min(1.0, lie_rate))
        # 允许注入确定性 rng（测试 / 复现用），默认用标准库随机
        self._rng = rng if rng is not None else __import__("random").Random()

    def answer(self, question, context):
        resp = super().answer(question, context)
        ans = resp.get("answer", "irrelevant")
        if ans not in ("yes", "no"):
            return resp  # 非 yes/no 不撒谎
        if self._rng.random() < self.lie_rate:
            flipped = "no" if ans == "yes" else "yes"
            return {"answer": flipped, "lied": True}
        return {"answer": ans, "lied": False}
