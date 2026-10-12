# 技术架构与目录结构

## 目录树（目标态）

```
AI-Arena/
├── .env                      # API key（不写死在代码）
├── .env.example              # key 模板
├── config.toml               # 模型注册表 + 对局参数
├── requirements.txt
├── engine/
│   ├── game.py               # 游戏状态机：回合/角色/胜负判定
│   ├── puzzles.py            # 题库（汤面/汤底/关键线索）
│   └── roles.py              # 角色定义（GM/提问者/骗子）
├── players/
│   ├── base.py               # Player 抽象接口（ask/answer/debate）
│   ├── llm_player.py         # litellm 统一适配
│   └── mock_player.py        # 离线假玩家（M0 先跑通，零 key）
├── prompts/                  # 各角色 system prompt
├── run/
│   ├── single_game.py        # 单模型单局
│   └── tournament.py         # 多模型循环对战
├── eval/
│   ├── metrics.py            # 四维度指标计算
│   └── report.py             # 图表(matplotlib)+文字结论
├── logs/                     # 对局 JSON 日志
├── analysis/                 # 评测报告输出（含图，便于发小红书）
└── doc/                      # 项目文档（本目录）
```

## 模块职责

| 模块 | 职责 | 依赖 |
|---|---|---|
| `engine/game.py` | 维护对局状态：当前回合、角色、已揭示线索、胜负判定；驱动回合循环 | 无（纯逻辑） |
| `engine/puzzles.py` | 题库数据源：每条含 汤面/汤底/关键事实集/线索梯度 | 无 |
| `engine/roles.py` | 角色契约：GM（守汤底答是非）、提问者（推理）、骗子（说谎/隐瞒） | 无 |
| `players/base.py` | 抽象 `Player`：输入观察 → 输出动作（统一 JSON 结构） | 无 |
| `players/llm_player.py` | 用 litellm 调模型，解析结构化输出，统计 token/延迟/成本 | litellm |
| `players/mock_player.py` | 规则化假玩家，保证零 key 也能跑通 pipeline | 无 |
| `run/single_game.py` | 编排一局：初始化 → 回合循环 → 落日志 | engine, players |
| `run/tournament.py` | 多模型配置化循环对战，汇总日志 | run/single_game |
| `eval/metrics.py` | 读日志算四维度指标 | pandas |
| `eval/report.py` | 出图（matplotlib）+ 文字结论 | pandas, matplotlib |

## 数据流

```
题库(puzzles.py) ──┐
                   ├─→ game.py(状态机) ─→ 每回合 ─→ Player.act(observation)
config.toml ───────┘                    ↑                │
                                        │                ↓
                                  胜负判定        litellm / mock
                                                        │
                                                        ↓
                                               logs/*.json（对局日志）
                                                        │
                                                        ↓
                                            eval/metrics.py → eval/report.py → analysis/
```

## 关键设计决策

1. **结构化动作**：所有 Player 输出统一为 JSON（如 `{"action":"ask","content":"他喝的是水吗？"}`），便于状态机解析、日志化、后续评测。
2. **mock 优先**：`mock_player` 保证 M0 零 key 跑通整条 pipeline，真实 key 到位后无缝切换。
3. **可观测性内建**：每次 LLM 调用记录 token / 延迟 / 成本，本身也是评测维度（效率/性价比）。
4. **降级策略**：某模型超时/不可用 → 自动 fallback 到 mock，保证对局不中断。
