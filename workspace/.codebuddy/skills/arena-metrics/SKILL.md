---
name: arena-metrics
description: "解释与复算 AI-Arena 推理指标：还原度 restoration / 汤底保真 guess_score / 效率 efficiency / 声明率，定位 eval/metrics.py 与 engine/game.py 的实现。关键词：指标、还原度、guess_score、efficiency、怎么算、可信、metrics"
version: "1.0.0"
author: "AI-Arena"
created: "2026-07-10"
updated: "2026-07-10"
---

# arena-metrics · 推理指标

> **Goal**: 讲清 AI-Arena 的评测指标「是什么、怎么算、为什么可信」，并给出复算入口。

## When to Use

- 用户问「还原度怎么算」「guess_score 是什么」「为什么指标可信」
- 用户怀疑指标有问题（真实模型还原度低，想区分是模型差还是 GM 容错不够）
- 要在报告/文档里解释指标定义

## 已实现指标（M2，规则+日志，零额外 LLM）

| 指标 | 字段 | 定义 | 含义 |
|---|---|---|---|
| 还原度 | `restoration` | 被 GM 确认(yes)的关键事实数 / 总关键事实数 | **推理探测过程** |
| 汤底保真 | `guess_score` | 声明汤底与真实汤底的 bigram Jaccard | **最终陈述**相似度 |
| 效率 | `efficiency` | `restoration / turns` | 推理经济性 |
| 命中事实 | `key_facts_hit` | 实际命中的关键事实列表 | 可解释性 |
| 声明率 | `declared` | 是否最终 guess | 是否收敛 |

## 为什么可信（关键）
- **SmartGM 关键词感知**（`players/gm.py`）：用 `keywords`/`clue_keywords` 做重叠匹配，容错模型自由改写（「他打嗝了吗」→ 命中「男人打嗝」）。还原度反映「是否探测到事实」，而非「是否逐字复制」。
- **过程/结果分离**：`restoration` 与 `guess_score` 分开，避免「胡猜蒙对」被误判强。
- **可信度自检**：强提问者（逐事实问）还原度≈1.0，弱提问者（问无关）≈0.0；拉不开说明指标/GM 有问题。

## 复算入口
```python
# eval/metrics.py
from eval.metrics import guess_score, load_logs, aggregate_by_model
guess_score("声明汤底", "真实汤底")          # 单局汤底保真
# engine/game.py :: Game._judge()            # restoration 计算
# engine/game.py :: Game._guess_score()      # guess_score 计算
```

## 阅读报告
- `analysis/metrics.csv`：每模型一行，含 `restoration_mean`/`guess_score_mean`/`efficiency_mean`；
- `analysis/restoration.png`：中文柱状图；
- `analysis/report.txt`：中文排序结论。

## 详见
- `doc/EVALUATION.md`（指标定义 + 四维度蓝图）
- `doc/METHODOLOGY.md` §6（让评测可信的工程要点）
