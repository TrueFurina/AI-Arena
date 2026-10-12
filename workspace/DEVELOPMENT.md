# 开发指南（DEVELOPMENT）

> 环境搭建与运行说明。规划背景见 `doc/` 与 `specs/engine-design.md`。

## 1. 环境要求

- **Python 3.11+**（本机实测 3.11.1；spec 目标 3.13，功能兼容）
- 包管理：pip（推荐虚拟环境，避免污染系统 Python）

## 2. 安装依赖

```bash
cd AI-Arena
python -m venv .venv && source .venv/bin/activate   # 可选但推荐
pip install -r requirements.txt
```

依赖：`litellm` / `pandas` / `matplotlib` / `python-dotenv` / `pydantic`。

## 3. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入对应供应商的 API Key（仅用真实模型时需要）
```

> 规则 2：key 只放 `.env`，不要写死在代码里。`.env` 不入库。

## 4. 运行

### 4.1 Mock 模式（无需任何 key，先跑通）

```bash
python run/single_game.py --mode mock
```

使用 `mock_player` 同时充当 GM 与提问者，跑完整条 pipeline 并产出 `logs/*.json`。

### 4.2 真实模型（需填 .env）

```bash
# 单模型单局（M1）。--model 支持友好别名或完整 litellm 名：
python run/single_game.py --mode llm --model deepseek          # → deepseek/deepseek-chat
python run/single_game.py --mode llm --model qwen              # → qwen/qwen-plus
python run/single_game.py --mode llm --model kimi              # → moonshot/moonshot-v1-8k
python run/single_game.py --mode llm --model "deepseek/deepseek-chat"   # 完整名亦可

# 多模型对战（M3）
python run/tournament.py --models deepseek qwen kimi
```

> **别名表**：`deepseek` / `qwen` / `kimi` / `gpt`(`openai/gpt-4o-mini`) / `claude`(`anthropic/claude-3-5-sonnet-20241022`)。
> key 由 litellm 按环境变量自动读取（`DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` / `MOONSHOT_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`），代码不写死。

### 4.3 预览提示词（无需 key）

```bash
# 只打印将发给模型的 system+user 消息，不发任何请求，便于先调提示词
python run/single_game.py --mode llm --model deepseek --puzzle turtle_001 --dry-run
```

### 4.4 选择主持人（GM）

```bash
# --gm smart（默认）：关键词感知，容错自由改写，还原度更可信（M2 推荐）
# --gm mock：逐字匹配（M0 旧行为，仅用于对照）
python run/single_game.py --mode mock --gm smart --puzzle turtle_001
```

### 4.5 生成评测报告（M2，无需 key）

```bash
# 先把若干对局日志落到 logs/，再聚合出指标与图表
python run/single_game.py --mode mock --gm smart --puzzle turtle_001 --log-dir logs
python run/evaluate.py --log-dir logs --out analysis
# 产出：analysis/metrics.csv（明细） + report.txt（文字结论） + restoration.png（图表）
```

### 4.4 M1 冒烟测试（无需 key，验证整条链路）

```bash
# 用假模型注入跑完整局，验证 LLMQuestioner → Game → 日志 全链路与 specs §7 结构
python tests/test_m1_smoke.py
```

> ⚠️ **M1 已知限制（非 bug）**：M1 的 GM 仍是规则化 `MockGM`，它只能对"提问里出现关键事实原文"的问题回 `yes`。真实模型的自由表述很难逐字命中原文，因此 M1 跑真实模型时 `restoration`（还原度）普遍偏低甚至为 0——这不代表模型推理差，只是 GM 与指标还不够聪明。**真正有意义的还原度/推理指标要到 M2（更聪明的 GM + 结构化指标）才成立。**

### 4.3 生成评测报告（M4）

```bash
python eval/report.py --log-dir logs --out analysis/
```

## 5. 项目结构速览

```
AI-Arena/
├── engine/       游戏引擎（game.py 状态机 / puzzles.py 题库 / roles.py 角色）
├── players/      玩家适配层（base / llm_player / mock_player / gm[SmarterGM]）
├── prompts/      各角色 system prompt
├── run/          编排脚本（single_game / evaluate / tournament）
├── tests/        冒烟测试（test_m1_smoke.py / test_m2_smoke.py，无需 key）
├── eval/         评测（metrics.py 算指标 / report 出图）
├── logs/         对局 JSON 日志
├── analysis/     评测报告输出
├── doc/          项目规划文档（6 篇）
├── specs/        技术规格（engine-design.md）
└── .workbuddy/   进展记忆（memory/MEMORY.md，规则 3）
```

## 6. 已知问题 / 注意事项

- ⚠️ **项目盘上传受限**：当前环境项目盘（tdrive）`file_upload` 接口存在 `file_size` 校验 bug，无法把文件传进项目盘 AI-Arena。文档与代码现阶段落本地工作目录，待接口修复后同步。
- ⚠️ **系统 Python 受 PEP 668 保护**：直接 `pip install` 会被拒。请使用虚拟环境（见 §2），或在自建环境用 `pip install --break-system-packages`（有风险，不推荐用于系统 Python）。本仓库验证用的隔离环境为 `/root/venv_aiarena`。
- 评测报告图表默认输出到 `analysis/`，便于发小红书 / 比赛（规则 5）。
- 遵循"先跑通再优化"（规则 4）：M0 先用 mock 跑通，再逐步接入真实模型与多角色变体。
