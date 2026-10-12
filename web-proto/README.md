# AI Arena · 海龟汤 AI 对战评测

让不同的 LLM 在海龟汤（情境推理）游戏里当玩家相互对抗，通过对局数据评估各模型的
推理 / 欺骗 / 伪装 / 辩论能力。本质是 **AI vs AI 的离线评测系统**。

## 架构

```
AI-Arena/
├── engine/            # 海龟汤对局引擎（无外部依赖）
│   ├── game.py        #   Puzzle / Message / GameResult + TurtleSoupGame 主循环
│   └── puzzles.py     #   题库加载
├── players/           # 玩家适配层
│   ├── base.py        #   Player / Host / Judge 接口
│   ├── mock.py        #   离线 Mock 玩家（零依赖跑通流程）
│   └── llm.py         #   litellm 统一接口（按 provider 前缀读 .env key）
├── analysis/
│   └── report.py      # 评分柱状图 + 文字结论（matplotlib）
├── puzzles.json       # 题库
├── run_game.py        # 命令行跑单局
├── web_app.py         # Web 评测界面（端口 8860）
└── logs/              # 对局 JSON 日志 + 报告（自动生成）
```

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env      # 填入各家 API Key

# 1) 命令行：纯 Mock 离线跑通（零依赖）
python run_game.py

# 2) 命令行：两个真实模型对战
python run_game.py --puzzle turtle_soup_001 \
    --questioners zai/glm-4.5-flash,deepseek/deepseek-chat \
    --host zai/glm-4.7 --judge openai/gpt-4o-mini --rounds 6

# 3) Web 界面
python web_app.py        # 打开 http://localhost:8860
```

## 模型名格式

`provider/model`（litellm 风格）。已支持的 provider → 环境变量：

| 前缀 | 环境变量 |
|------|---------|
| `zai/` | `ZAI_API_KEY` |
| `moonshot/` | `MOONSHOT_API_KEY` |
| `deepseek/` | `DEEPSEEK_API_KEY` |
| `openai/` | `OPENAI_API_KEY` |
| `qwen/` | `DASHSCOPE_API_KEY` |
| `anthropic/` | `ANTHROPIC_API_KEY` |
| `gemini/` | `GOOGLE_API_KEY` |
| `ashbwin` | `ASHBWIN_API_KEY` + `ASHBWIN_API_BASE` |

以 `mock-` 开头的模型为内置离线模型，开箱即用（先跑通流程）。

## 接入免费端点 ashbwin

见 `arena_providers.py`：把它注册成 litellm 自定义模型，页面填 `ashbwin` 即可。
（注意：该端点目前仅在特定网络可达，详见对接记录。）

## 下一步

- [ ] 扩展到多模型自动对战 + 批量评测
- [ ] LLM 裁判的评分一致性校准
- [ ] 狼人杀等更多社交推理游戏
