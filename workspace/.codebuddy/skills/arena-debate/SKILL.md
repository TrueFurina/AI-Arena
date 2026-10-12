---
name: arena-debate
description: "M6 辩论 + LLM-as-Judge：正反方多轮对抗，结构化裁判 verdict 衡量说服力。零 key 可运行演示，真实 LLM 接入点已用 completion_fn 预留。关键词：辩论、debate、裁判、judge、说服力、对抗、M6"
version: "1.0.0"
author: "AI-Arena"
created: "2026-07-17"
updated: "2026-07-17"
---

# arena-debate · 辩论 + 裁判（辩论维度）

> **Goal**: 用 `run/debate_demo.py` 演示 M6 —— 正反方多轮辩论，MockJudge 给出结构化 verdict（winner / scores / rationale）。

## When to Use

- 用户说「加辩论」「让模型互怼」「LLM 当裁判」「测说服力」
- 想从「单人推理」升维到「多 Agent 立场对抗」
- 要扩展指标到 persuasion / consensus 维度

## 用法

### 零 key 演示
```bash
python run/debate_demo.py                 # 默认命题 + 3 轮
python run/debate_demo.py --rounds 4 --proposition "AI 应被允许自主参与海龟汤评测"
# 输出逐轮发言 + 裁判结论（winner / scores / rationale）+ 说服力分差
```

### 在代码里用框架
```python
from engine.debate import Debate, MockDebater, MockJudge
debate = Debate(
    proposition="...",
    debaters={"pro": MockDebater("pro", PRO_ARGS), "con": MockDebater("con", CON_ARGS)},
    judge=MockJudge(),
    rounds=3,
)
verdict = debate.run()
print(verdict)                            # {'winner','scores','rationale'}
```

## 指标（详见 specs/debate-design.md）
| 指标 | 含义 |
|---|---|
| `persuasion_score` | 单方裁判得分（证据密度 + 立场一致性 + 篇幅） |
| `persuasion_margin` | `|pro - con|`，分差越大立场优势越明显 |

## 架构要点
- `engine/debate.py`：`Debater` / `MockDebater` / `Judge` / `MockJudge` / `Debate` 状态机。
- **依赖注入**：真实 LLM 接入点用 `completion_fn` 预留（与 LLMQuestioner 同范式）；
  零 key 用 Mock 跑通，拿到 key 后换 `LLMDebater` / `LLMJudge`，接口不变。
- `MockJudge` 用透明启发式（证据词命中 + 篇幅），非黑箱，便于审计。

## 注意事项
- 当前 PoC 是零 key 框架 + Mock 裁判。真实 `LLMJudge` 必须明示评分 rubric（逻辑/证据/切题），
  并多次复跑取 consensus、随机化发言顺序，缓解 LLM 裁判偏见（见设计规格 §5）。
- 真实 key 就绪前先用 Mock 验证 pipeline，不要过早投资 LLMDebater（先度量后优化）。
