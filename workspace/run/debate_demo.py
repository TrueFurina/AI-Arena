"""M6 辩论 + LLM-as-Judge · 零 key 可运行演示（PoC）。

证明「辩论维度」能落地并产出结构化裁判 verdict，不依赖任何 API key。

做了什么：
- 给定命题，正反两方（MockDebater）按预置论点多轮交锋；
- MockJudge 用透明启发式（证据密度 + 篇幅）给双方打分，判出 winner 与分差；
- 打印逐轮发言 + 裁判结论 + 说服力分差指标。

为什么是 PoC（见 specs/debate-design.md）：
- 真实 LLM 接入点已通过依赖注入 completion_fn 预留（与 LLMQuestioner 同范式）：
  换 LLMDebater / LLMJudge 即可用真实模型，但零 key 下先用 mock 验证 pipeline。
- 更丰富的裁判维度（逻辑一致性、是否切题、是否被对方反驳动摇）留给后续。

用法：
    python run/debate_demo.py                 # 默认命题 + 3 轮
    python run/debate_demo.py --rounds 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.debate import Debate, MockDebater, MockJudge


# 零 key 演示用的预置论点池（真实场景由 LLM 生成）
PRO_ARGS = [
    "数据显示，AI 在结构化推理任务上的复现误差已低于人类标注员，因此可用于海龟汤评测。",
    "例如把 GM 判定做成关键词感知后，跨 8 题的还原度指标可复现，证明评测可信。",
    "因为评测全程离线、零人工干预，AI 自主参与反而比人工更可审计。",
    "对比人工出题，AI 能在 30 秒内生成 100 组对局，效率优势明显。",
]
CON_ARGS = [
    "研究指出，大模型存在幻觉，自主参与可能污染评测结论，需要人类把关。",
    "例如当 GM 撒谎时，弱模型鲁棒性仅 0.25，说明 AI 未必能识别误导。",
    "因为可解释性不足，AI 给出的推理过程难以被复核，存在黑箱风险。",
    "对比人工评审，AI 缺少对语境的常识判断，容易误判边界案例。",
]


def main():
    parser = argparse.ArgumentParser(description="M6 辩论零 key 演示")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--proposition", default="AI 应被允许自主参与海龟汤评测")
    args = parser.parse_args()

    debate = Debate(
        proposition=args.proposition,
        debaters={
            "pro": MockDebater("pro", PRO_ARGS),
            "con": MockDebater("con", CON_ARGS),
        },
        judge=MockJudge(),
        rounds=args.rounds,
    )

    print(f"命题：{args.proposition}\n{'=' * 50}")
    verdict = debate.run()
    for t in debate.transcript:
        tag = "✅ 正方" if t["stance"] == "pro" else "❌ 反方"
        print(f"[第{t['round']}轮] {tag} {t['speaker']}：{t['text']}")

    print("\n" + "=" * 50)
    print("🏛️ 裁判结论（MockJudge 启发式）")
    print(f"  胜方：{verdict['winner']}")
    print(f"  各得分：{verdict['scores']}")
    print(f"  说明：{verdict['rationale']}")

    scores = verdict["scores"]
    if "pro" in scores and "con" in scores:
        margin = abs(scores["pro"] - scores["con"])
        print(f"  说服力分差（persuasion_margin）：{margin:.3f}")
    print("\n解读：分差越大代表立场优势越明显；真实 LLMJudge 会额外评估逻辑一致性与切题度。")


if __name__ == "__main__":
    main()
