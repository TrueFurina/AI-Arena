---
name: arena-report
description: "把 analysis/metrics.csv 翻译成可发小红书 / 可投稿比赛的可读战报（analysis/SHARE.md + 刷新 restoration.png）。关键词：战报、小红书、比赛、分享、report、SHARE、可视化、文案、M4"
version: "1.0.0"
author: "AI-Arena"
created: "2026-07-17"
updated: "2026-07-17"
---

# arena-report · 可读战报生成

> **Goal**: 用 `run/report_share.py` 把工程态聚合产物（CSV/PNG）变成带战力榜、各题矩阵、现成文案提示的 Markdown，直接贴小红书或塞进比赛报告。

## When to Use

- 用户说「出一份能发小红书的战报」「做个比赛用的报告」「把评测结果排版好」
- 已有 `analysis/metrics.csv`（跑过 tournament / evaluate 后）
- 要让评测结果「人读得懂、能传播」

## 用法

### 一键生成战报
```bash
python run/report_share.py
# 读 analysis/metrics.csv
# 写 analysis/SHARE.md（战力榜 + 指标速读 + 一句话结论 + 还原度图 + 各题矩阵 + 文案提示）
# 刷新 analysis/restoration.png
```

### 指定输入输出
```bash
python run/report_share.py --csv analysis/metrics.csv --out analysis/SHARE.md
python run/report_share.py --chart analysis/restoration.png
```

## SHARE.md 结构
1. **🏆 模型战力榜** — 按均值还原度排名，带奖牌图标
2. **📖 指标速读** — `<details>` 折叠，解释 restoration / guess_score / efficiency
3. **📌 一句话结论** — 直接可摘抄的传播句
4. **📊 可视化** — 还原度柱状图引用
5. **🧩 各题表现** — 模型 × 题目的均值还原度矩阵
6. **📱 小红书文案提示** — 标题/钩子/标签/互动尾句
7. **🏅 比赛/技术报告文案提示** — 问题定义/方法亮点/结论句式/可复现声明

## 设计纪律（见 doc/METHODOLOGY.md）
- **零 key 优先**：只读 CSV，绝不触发任何 LLM 调用。
- **先度量后优化**：所有数字都来自 tournament 真实落盘的指标，不编造。
- **配置/代码分离**：模型名、阈值都来自数据，脚本只负责排版。

## 注意事项
- 数据源必须是 `run/evaluate.py` 或 `run/tournament.py` 产出的 `metrics.csv`（列：model,puzzle_id,restoration,guess_score,efficiency,turns_used,declared,n_facts,n_hit）。
- 无 matplotlib 时图表跳过，但 SHARE.md 仍会生成，文案与表格完整可用。
- 想换文案风格：直接改 `build_markdown()` 里的小红书 / 比赛提示段，不影响数据逻辑。
