---
name: arena-tournament
description: "运行 AI-Arena M3 多模型对战：读 config.toml 模型注册表，对每个 enabled 模型 × 每题跑 N 局（vs SmartGM），自动聚合出评测报告。零 key 可用 mock 基线验证编排。关键词：多模型、对战、tournament、config.toml、注册表、批量、M3"
version: "1.0.0"
author: "AI-Arena"
created: "2026-07-13"
updated: "2026-07-13"
---

# arena-tournament · 多模型对战

> **Goal**: 用 `run/tournament.py` 按 `config.toml` 注册表批量跑多模型对战，闭环产出报告。

## When to Use

- 用户说「跑多模型对战」「批量评测」「对比所有模型」「按注册表跑」
- 已配好 `config.toml` 的 `[[models]]`，想一键出跨模型数据
- 想零 key 验证编排（mock 基线）

## 用法

### 一键按注册表跑全部 enabled 模型
```bash
python run/tournament.py
# 读 config.toml -> 每 enabled 模型 × 每题 runs_per_puzzle 局 -> logs/
# evaluate_after=true 时自动调用 evaluate.py 出 analysis/ 报告
```

### 零 key 验证编排
```bash
python run/tournament.py --mock
# 真实模型也临时用 mock-strong 跑，不依赖任何 key，验证循环不崩
# 预期：mock-strong 还原度≈1.0，mock-weak≈0.0（指标区分度正确）
```

### 精控
```bash
python run/tournament.py --model deepseek --puzzle turtle_001 --runs 2
python run/tournament.py --no-evaluate        # 跑完不自动聚合
python run/tournament.py --config my.toml     # 指定注册表
```

## 注册表（config.toml）
```toml
[globals]
max_turns = 20
runs_per_puzzle = 2
gm = "smart"
log_dir = "logs"
evaluate_after = true

[[models]]
alias = "deepseek"
litellm = "deepseek/deepseek-chat"
enabled = false          # 填好 .env key 后改 true
note = "DeepSeek"
[[models]]
alias = "mock-strong"
mock = "strong"
enabled = true           # 零 key 立即可跑
```

## 输出
- 每局：`logs/<model>_<puzzle>_<ts>.json`
- 聚合：`analysis/metrics.csv` + `report.txt` + `restoration.png`（由 evaluate.py 生成）

## 注意事项
- 真实模型条目需先在 `.env` 填对应 key 且 `enabled=true` 才参与；
- `evaluate.py` 聚合 `logs/` 下**所有**日志；限定范围可指定独立 `log_dir` 或先清空旧日志；
- 设计细节见 `specs/tournament-design.md`。
