---
name: arena-run
description: "运行 AI-Arena 海龟汤对局（单局/批量），支持零 key 的 mock 模式与真实 LLM（litellm）模式，含 --dry-run 提示词预览。关键词：跑对局、单局、海龟汤、single_game、llm、mock、dry-run、真实模型"
version: "1.0.0"
author: "AI-Arena"
created: "2026-07-10"
updated: "2026-07-10"
---

# arena-run · 运行对局

> **Goal**: 用 `run/single_game.py` 跑一局海龟汤，产出结构化 JSON 日志到 `logs/`。

## When to Use

- 用户说「跑一局」「测试某个模型」「用 deepseek 跑 turtle_001」「预览提示词」
- 用户想验证新模型/新题目/新 GM 行为
- 用户要产出可用于 `arena-evaluate` 的对局日志

## 用法

### 零 key · mock 模式（规则化假玩家，无需 API key）
```bash
python run/single_game.py --mode mock --puzzle turtle_001
```

### 真实 LLM 模式（需先在 .env 填 key，见 doc/KEYS.md）
```bash
python run/single_game.py --mode llm --model deepseek --gm smart --puzzle turtle_001
# --model 别名: deepseek / qwen / kimi / gpt / claude，或完整 litellm 名
# --gm: smart(关键词感知,默认) / mock(逐字匹配)
```

### 仅预览提示词（不发请求，零 key 也能用）
```bash
python run/single_game.py --mode llm --model deepseek --gm smart --dry-run
```

### 关键参数
| 参数 | 说明 |
|---|---|
| `--mode` | `mock`（零 key） / `llm`（真实模型） |
| `--model` | llm 模式下的模型别名或完整 litellm 名 |
| `--gm` | `smart`（默认，关键词感知） / `mock`（逐字） |
| `--puzzle` | 题目 id（如 turtle_001），缺省随机 |
| `--max-turns` | 最大轮数，默认 20 |
| `--dry-run` | 只打印提示词与用户消息，不调用 API |
| `--log-dir` | 日志目录，默认 `logs` |

## 输出
- 日志：`logs/<model>_<puzzle>_<ts>.json`（结构见 `specs/engine-design.md` §7）
- 终端打印：模型 / 题目 / 轮数 / 还原度 / 汤底保真 / 命中关键事实

## 注意事项
- 真实调用前先用 `arena-...` 的 test_api 或 `python run/test_api.py` 验证连通性。
- 无 key 跑 llm 模式会快速失败并退出码 1（友好提示），不会假装成功。
- 接新供应商前确认其 `chat/completions` 可达（tokenc.top 教训：通用聚合网关未必提供 LLM）。
