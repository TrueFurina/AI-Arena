"""M2 冒烟测试（无需 API key）。

验证目标：
1. SmartGM 容错改写：同一句自由表述提问，MockGM 答 irrelevant，SmartGM 答 yes。
2. 区分度：SmartGM 下，强提问者（问到每个关键事实）还原度≈1.0，弱提问者（问无关问题）≈0。
   若指标能拉开差距，说明还原度确实反映推理能力，而非逐字命中运气。
3. 指标模块：聚合多局、生成图表、日志含 guess_score/efficiency 新字段。
运行：/root/venv_aiarena/bin/python tests/test_m2_smoke.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import Game
from engine.puzzles import get_puzzle, PUZZLES
from players.mock_player import MockGM, MockQuestioner
from players.gm import SmartGM
from players.base import Player
from eval import metrics


class WeakQuestioner(Player):
    """弱提问者：问与真相无关的泛泛问题，然后直接猜底。用于证明指标能区分强弱。"""
    name = "weak"

    def __init__(self, bottom, n=3):
        self.bottom = bottom
        self.pool = ["今天的天气好吗？", "你喜欢猫吗？", "现在是几月？", "你吃过饭了吗？"]
        self.n = n
        self.i = 0

    def act(self, observation):
        if self.i < self.n:
            q = self.pool[self.i % len(self.pool)]
            self.i += 1
            return {"action": "ask", "content": q}
        return {"action": "guess", "content": self.bottom}


def _ctx(puzzle):
    return {
        "puzzle": puzzle,
        "key_facts": puzzle["key_facts"],
        "clue_gradient": puzzle["clue_gradient"],
        "history": [],
    }


def test_gm_smarter():
    p = get_puzzle("turtle_001")
    q = "他是不是打嗝了？"  # 自由改写，不含事实原文「男人打嗝」
    assert MockGM().answer(q, _ctx(p))["answer"] == "irrelevant"
    assert SmartGM().answer(q, _ctx(p))["answer"] == "yes"
    print("✅ SmartGM 容错改写：同一句 MockGM=irrelevant，SmartGM=yes")


def test_discrimination():
    strong, weak = [], []
    for p in PUZZLES:
        g = SmartGM()
        sg = Game(p, g, MockQuestioner(p["key_facts"], p["bottom"]),
                  max_turns=20, log_dir=tempfile.mkdtemp())
        sg.run()
        strong.append(sg.build_log()["result"]["restoration"])

        wg = Game(p, g, WeakQuestioner(p["bottom"]),
                  max_turns=20, log_dir=tempfile.mkdtemp())
        wg.run()
        weak.append(wg.build_log()["result"]["restoration"])

    sr = sum(strong) / len(strong)
    wr = sum(weak) / len(weak)
    assert sr >= 0.9, f"强提问者还原度应≥0.9，实际 {sr}"
    assert wr <= 0.1, f"弱提问者还原度应≤0.1，实际 {wr}"
    assert sr > wr
    print(f"✅ 区分度：强提问者均值还原度 {sr:.2f} vs 弱提问者 {wr:.2f}")


def test_metrics_and_log_fields():
    d = tempfile.mkdtemp()
    for p in PUZZLES:
        g = SmartGM()
        game = Game(p, g, MockQuestioner(p["key_facts"], p["bottom"]),
                    max_turns=20, log_dir=d)
        game.run()
        game.save()

    logs = metrics.load_logs(d)
    assert len(logs) == len(PUZZLES)
    r0 = logs[0]["result"]
    for fld in ("guess_score", "efficiency"):
        assert fld in r0, f"日志缺字段 {fld}"

    rows = metrics.summarize(logs)
    agg = metrics.aggregate_by_model(rows)
    assert "mock" in agg
    assert agg["mock"]["mean_restoration"] >= 0.9

    chart = metrics.make_chart(agg, os.path.join(d, "restoration.png"))
    assert chart and os.path.exists(chart), "图表未生成"
    report = metrics.text_report(agg)
    assert "结论" in report

    print(f"✅ 指标模块：{len(logs)} 局聚合成功，图表 {chart}")
    print(f"   均值还原度 {agg['mock']['mean_restoration']:.2f} | "
          f"汤底保真 {agg['mock']['mean_guess']:.2f} | "
          f"效率 {agg['mock']['mean_efficiency']:.3f}")


if __name__ == "__main__":
    test_gm_smarter()
    test_discrimination()
    test_metrics_and_log_fields()
    print("\n✅ M2 冒烟测试全部通过")
