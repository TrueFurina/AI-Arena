---
name: arena-evaluate
description: "聚合 logs/ 下的对局 JSON 日志，生成跨模型评测报告（analysis/metrics.csv + report.txt + restoration.png）。关键词：评测、聚合、报告、evaluate、metrics、对比、还原度、横向比较"
version: "1.0.0"
author: "AI-Arena"
created: "2026-07-10"
updated: "2026-07-10"
---

# arena-evaluate · 聚合评测

> **Goal**: 用 `run/evaluate.py` 把多局日志聚合成可比较的跨模型报告。

## When to Use

- 用户说「出评测报告」「对比几个模型」「聚合 logs」「生成图表」
- 已有若干 `logs/*.json`，想看哪个模型还原度高、效率高
- 要产出发小红书/比赛的可读报告

## 用法

### 一键聚合
```bash
python run/evaluate.py
# 读 logs/ -> 产出到 analysis/：
#   metrics.csv      跨模型聚合（均值还原度/汤底保真/效率/轮次/声明率）
#   report.txt       中文文字结论（按还原度排序）
#   restoration.png  中文分组柱状图
```

### 指标复算 / 自定义分析
直接调用 `eval/metrics.py`：
```python
from eval.metrics import load_logs, aggregate_by_model, make_chart, text_report
logs = load_logs("logs")
agg  = aggregate_by_model(logs)      # DataFrame: 每模型一行
make_chart(agg, "analysis/restoration.png")
print(text_report(agg))
```

## 指标含义（详见 doc/EVALUATION.md）
| 指标 | 含义 |
|---|---|
| `restoration` | 还原度：被 GM 确认的关键事实占比（推理过程） |
| `guess_score` | 汤底保真：声明汤底与真实汤底的 bigram Jaccard（结果相似度） |
| `efficiency` | 效率：restoration / turns（推理经济性） |
| `declared_rate` | 声明率：最终是否 guess |

## 报告怎么读
- `metrics.csv`：每行一个模型，横向比强弱；
- `restoration.png`：分组柱状图，一眼看谁还原度高；
- `report.txt`：中文排序结论，可直接摘抄。

## 注意事项
- 旧日志缺新字段时指标模块补 0，不崩（向后兼容）。
- 图表用 matplotlib，自动注册 Noto CJK；无中文字体时退化为英文标签（不掩盖真实问题）。
- 想让指标可信：先用 SmartGM（`--gm smart`）且题库配好 `keywords`，否则还原度偏低是 GM 容错问题不是模型问题。
