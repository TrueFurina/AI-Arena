# AI-Arena 编程方法论

> 写给未来接手或协作本项目的开发者（也包括未来的我自己）。
> 本文件定义「在这个项目里应该怎么写代码、怎么扩展、怎么不踩坑」。
> 它不是一次性规划，而是**长期有效的工程约定**。

---

## 0. 一句话哲学

> **先跑通，再优化；用结构化日志把「模型的推理能力」变成可观测、可比较、可复现的数据。**

AI-Arena 本质是 **AI-vs-AI 离线评测系统**，不是真人游戏产品。我们衡量的是「不同模型在社交推理游戏里的能力差异」，所以**可复现的对局数据**比花哨的 UI 重要得多。一切工程决策都服务于「让评测可信、可比较、可自动化」。

---

## 1. 核心原则（带 SRE 视角）

把 SRE 的工程纪律平移到评测系统上，会得到一套非常顺手的原则：

### 1.1 零 key 优先（Zero-key-first）
任何核心链路都必须能在**没有任何 API key** 时跑通。
- 玩家/GM 必须有 `mock` 实现（规则化、确定性），保证 pipeline 不卡在 key 上。
- 真实模型通过依赖注入接入（`LLMQuestioner(completion_fn=...)`），无 key 时用假模型注入做冒烟。
- 真实调用失败时**立即、清晰地报错并退出码 1**，绝不在无 key 时「假装成功」刷屏（那会污染评测数据）。

### 1.2 结构化日志即可观测性（Observability）
- 每一局产出一份 **结构化 JSON 日志**（`logs/<model>_<puzzle>_<ts>.json`），而非散落的 print。
- 日志字段是契约：见 `specs/engine-design.md` §7。新增字段要向后兼容（旧日志缺字段时指标模块补 0，不崩）。
- 日志 = 黄金信号来源：还原度、汤底保真、效率、命中事实都从日志算，不在运行时硬编码。

### 1.3 先度量，后优化（Measure before optimize）
- 没有指标证明「某个改动让推理评测更准」之前，不要动判定逻辑。
- 用 `run/evaluate.py` 聚合出 `analysis/metrics.csv` + `report.txt` + `restoration.png`，**先看懂数据再决定改什么**。
- 反例：看到真实模型还原度低就以为「模型差」——其实是 GM 逐字匹配容错不够（这正是 M2 引入 SmartGM 的原因）。

### 1.4 自动化 toil（Automate toil, don't heroic）
- 凡是「重复两次以上」的操作（跑对战、出报告、加题、查 key）都值得做成 **skill / 脚本**。
- 本仓库已沉淀 skills：`arena-run` / `arena-evaluate` / `arena-puzzle` / `arena-metrics`（见 `.codebuddy/skills/`）。
- 目标：未来跑一场 5 模型 × 8 题 × 3 局的对战，应该是一条命令，而不是手动敲 120 次。

### 1.5 无责复盘（Blameless）
- 模型「答错」「猜错汤底」**是评测数据，不是故障**。系统失败，不是人/模型失败。
- 报告里写「claude 在 turtle_003 还原度 0.4」，而不是「claude 不行」。
- 排查问题时改系统（GM 容错、指标定义），不指责模型输出。

### 1.6 渐进式接入（Progressive rollout）
新模型/新玩法上线走三级：
1. `mock`：规则化假玩家跑通整条 pipeline；
2. `--dry-run`：只打印发给真实模型的提示词，**不发请求**，先核对提示词与动作协议；
3. 真实调用：填 `.env` 后真实对局，用 `run/test_api.py` 先验证连通性再开跑。

### 1.7 配置与代码分离（规则 2）
- 所有 key / 端点 / 开关放 `.env`，**代码里绝不写死**。
- litellm 按环境变量自动读取 `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` / `OPENAI_API_KEY` 等。
- `.env` 不入库（gitignore）；`.env.example` 入库当模板。

### 1.8 知识沉淀（规则 3）
- 每次完成「大版本」（里程碑 / 重大重构 / 关键结论），更新 `.workbuddy/memory/MEMORY.md`。
- 记录「做了什么、现状、下一步、关键决策、开放问题」，而不是只写 changelog。

---

## 2. 里程碑驱动开发（M0–M6）

不要一上来做完美主义。按里程碑推进，**每个里程碑都可独立验证、可回滚**：

| 里程碑 | 内容 | 交付物 | 依赖 key |
|---|---|---|---|
| **M0 脚手架** | 引擎 + mock 玩家跑通一局 | 端到端可运行 demo | ❌ |
| **M1 单模型单局** | 真实 LLM 当提问者（GM 规则化） | 单局 JSON 日志 | ✅(真实调用) |
| **M2 题库 + 指标 v1** | 扩充题库、SmartGM、推理指标 | 单局评测分 + 报告 | ❌(代码已就绪) |
| **M3 多模型对战** | tournament 循环 + 模型注册表 | 多模型对比数据 | ✅(≥2 真实模型) |
| **M4 评测报告** | 图表 + 文字结论 | 可读报告（图+文） | — |
| **M5 骗子变体** | 角色化（诚实/说谎） | 评测欺骗/伪装维度 | — |
| **M6 辩论环节** | 分歧辩论 + LLM-as-Judge | 四维度全覆盖 | — |

> 起步任务四条（引擎+题库 / 适配层 / 单模型单局 / 多模型+评测）≈ **M0–M4**，是现在就要做的核心。

**推进纪律**：M 未完成前不偷偷开下一个 M 的大块头工作；脚手架（零 key）先于真实调用。

---

## 3. 代码组织约定

```
AI-Arena/
├── engine/            # 纯逻辑：状态机 + 回合循环 + 日志。不依赖具体模型
│   ├── roles.py       # 角色/答案词表（Answer 枚举）
│   ├── puzzles.py     # 题库（get_puzzle）
│   └── game.py        # Game 状态机 + _judge + build_log + save
├── players/           # 玩家适配层
│   ├── base.py        # Player / GM 抽象基类
│   ├── mock_player.py # 规则化假玩家（零 key）
│   ├── llm_player.py  # LLMQuestioner（litellm 适配，支持依赖注入）
│   └── gm.py          # SmartGM（关键词感知，零 key）
├── prompts/           # 可编辑的 system prompt（questioner.md）
├── eval/              # 指标与报告（metrics.py）
├── run/               # CLI 入口
│   ├── single_game.py # 单局编排（--mode mock/llm, --gm mock/smart, --dry-run）
│   ├── evaluate.py    # 评测聚合入口
│   └── test_api.py    # API 连通性自检
├── specs/             # 协议级规格（engine-design.md）
├── doc/               # 文档（本文件 + ROADMAP/DESIGN/...）
├── tests/             # 零 key 冒烟 + 假模型注入
├── logs/              # 对局 JSON 日志（运行产物）
├── analysis/          # 评测报告（运行产物）
├── .env / .env.example
└── .workbuddy/memory/MEMORY.md
```

约定：
- `engine/` 保持**纯逻辑**、少副作用，便于单测与注入。
- `players/llm_player.py` 只负责「把 observation 变成模型调用、把回复解析成动作」，不写判定逻辑。
- 判定/GMath 逻辑放 `engine/game.py` 与 `players/gm.py`，**GM 永不泄漏汤底**。
- 所有玩家输出统一为 JSON 动作协议（`{"action":"ask|guess","content":str}`），见 `specs/engine-design.md` §5.2。

---

## 4. 测试策略

不依赖外部 key 也能验证正确性：

1. **假模型注入（dependency injection）**：`LLMQuestioner(completion_fn=fake_fn)`。
   测试里用闭包计数器顺序消费预设动作，断言日志结构、`restoration`、`key_facts_hit`、围栏 JSON 可解析。
2. **零 key 冒烟**：`pytest tests/` 全绿 = 链路健康。无 key 时真实调用路径应快速失败退出码 1，而非假成功。
3. **向后兼容**：指标模块读取旧日志时，缺字段补 0，不抛异常。
4. **中文渲染**：matplotlib 注册 Noto CJK，无中文字体时退化为英文标签（避免 Glyph missing 警告掩盖真实问题）。
5. **tokenc 教训（固化）**：外部「聚合网关」未必提供 LLM；接入新供应商前先用 `run/test_api.py` 或最小 curl 验证 `chat/completions` 可达，再写进配置。

---

## 5. 密钥与运行环境

- **Python 环境**：系统 Python 受 PEP 668 保护时，建隔离 venv（如 `/root/venv_aiarena`）装 `litellm` / `pandas` / `matplotlib` / `python-dotenv` / `pydantic`，不要 `--break-system-packages`。
- **密钥**：只放 `.env`（规则 2）。详见 `doc/KEYS.md`。
- **自检**：`python run/test_api.py` 一键验证所有供应商连通性 + tokenc 网关。
- **自定义 OpenAI 兼容端点**：设 `OPENAI_API_BASE` + `OPENAI_API_KEY`，用 `--model openai/<模型名>` 即可，无需改码。

---

## 6. 让评测「可信」的工程要点

指标不可信 = 评测无意义。已落地的可信化手段：

- **SmartGM 关键词感知**（M2）：用每题预置 `keywords` / `clue_keywords` 做重叠匹配，容错真实模型的自由改写（「他打嗝了吗」→ 命中事实「男人打嗝」），让还原度反映「是否探测到事实」而非「是否逐字复制」。
- **还原度 ≠ 汤底保真**：
  - `restoration`（还原度）= 被 GM 确认(yes)的关键事实占比，衡量**推理探测过程**；
  - `guess_score`（汤底保真）= 声明汤底与真实汤底的 bigram Jaccard，衡量**最终陈述相似度**。
  - 两者分离，才能区分「会问问题但说不清」与「胡猜但蒙对」。
- **效率**：`efficiency = restoration / turns`，鼓励用更少轮次还原。
- **跨模型聚合**：`eval/metrics.aggregate_by_model` 按模型聚均值，产出可横向比较的表。

> 验证可信度的土办法：构造「强提问者」（逐事实提问）与「弱提问者」（问无关问题），在 SmartGM 下还原度应分别 ≈1.0 / ≈0.0。差距拉不开，说明指标或 GM 有问题。

---

## 7. 四维度与玩法演进

| 玩法 | 评测维度 | 如何评测 |
|---|---|---|
| A. 单 GM + 单提问者（当前） | **推理** | restoration / guess_score / efficiency |
| B. 双盲互猜 | 推理 + 伪装 | 双方互探，看谁先还原 |
| C. 骗子变体（诚实者/说谎者） | **欺骗 / 伪装** | 说谎者能否误导、诚实者能否被识破 |
| D. 辩论裁决 + LLM-as-Judge | **辩论** | 分歧点辩论，Judge 判胜负 |

演进顺序：先 A 跑通推理指标 → B/C 覆盖欺骗伪装 → D 用 LLM-as-Judge 做细粒度打分。Judge 本身也应是多模型交叉，避免单模型偏见。

---

## 8. 何时写 skill / doc

- **skill 化**：任何「重复两次以上」的运维/评测操作（跑对战、出报告、加题、查 key）做成 `.codebuddy/skills/<name>/SKILL.md`，正文含 `When to Use` + 精确命令/路径，未来一键调用。
- **doc 化**：关键决策、协议规格（`specs/`）、扩展方式（`doc/EXTENDING.md`）、密钥管理（`doc/KEYS.md`）、评测含义（`doc/EVALUATION.md`）都写下来，避免「只有当事人脑里有」。

---

## 9. 变更纪律（硬规则）

1. 所有代码/文件/日志放在 AI-Arena 目录下，**不往外写**（规则 1）。
2. key / 端点放 `.env`，**不写死进代码**（规则 2）。
3. 大版本更新 `MEMORY.md`（规则 3）。
4. 用户偏好「先跑通再优化」，**不要一上来做完美主义**（规则 4）。
5. 评测报告要可读（图表 + 文字结论），方便发小红书/比赛（规则 5）。

---

## 10. 常见反模式（踩坑清单）

- ❌ 无 key 时让真实调用「重试到成功」或假装成功 → 污染评测数据。
- ❌ 把「模型自由改写提问」当成「还原度低」→ 应先检查 GM 容错（SmartGM）。
- ❌ 在代码里硬编码 key / 模型名 → 违反规则 2，且无法多模型对比。
- ❌ 过度设计：还没跑通单局就先做 M6 辩论 → 违反「先跑通再优化」。
- ❌ 新供应商没验证 `chat/completions` 就写进配置 → tokenc 教训。
- ❌ 日志改字段不向后兼容 → 旧日志聚合崩。
- ❌ 指标没证明有效性就调判定逻辑 → 违反「先度量后优化」。

---

> 本文件与 `ROADMAP.md` / `EXTENDING.md` / `KEYS.md` / `EVALUATION.md` 互为补充：ROADMAP 管「往哪走」，EXTENDING 管「怎么加」，KEYS 管「怎么连」，EVALUATION 管「怎么信」，本文档管「怎么写」。
