# AI-Arena · 项目总览

AI vs AI 社交推理游戏评测台（海龟汤起步）：让不同 LLM 在**推理 / 欺骗 / 伪装 / 辩论**四个维度上对抗，对局过程落结构化 JSON 日志，用量化指标横向比较各模型能力。

**定位（2026-10-12 用户拍板）**：升级为**「主线 LLM + MARL agent 的链上可验证评测台」**，对齐多智能体主线四层（L1 LLM 决策 / L2 MARL 执行 / L3 链上可信 / L4 展示）。同类的剧本杀平台已放弃，本项为唯一主攻项目。后续路线见 [`ROADMAP.md`](ROADMAP.md)。

## 目录结构（三条开发线）

| 目录 | 路线 | 状态 |
|---|---|---|
| `web-proto/` | Flask 评测台（端口 8860） | 🔒 **唯一主线**（2026-10-13 定）：L1 多 LLM 对抗 + L2 RL/MARL 策略（P2-B B-1 可学习主持 + 对抗自对弈已实现，服务器首跑为诚实负结果）+ 指标口径单一真值 + held-out 泛化协议 + **L3 链上评测凭证（P1）与多裁判共识/身份锚定（P2-A）**；**pytest 102/102 全绿**；题库 12 题 |
| `mcp-server/` | MCP 服务器（FastMCP / stdio） | 🧊 **冻结归档**（2026-10-13 定）：v0.5 已完成、pytest 45/45，四性格玩家 / 12 题 / deception / challenge / debate 俱全。**不再新增功能**，其游戏机制概念已被主线吸收 |
| `workspace/` | 文档 / 技能 / M0–M6 脚手架线 | 📚 设计资产库：12 篇设计文档（`doc/`）+ 4 篇规格（`specs/`）+ 8 个项目技能（`.codebuddy/skills/`）+ tournament/evaluate/report_share 等编排脚本 |

## 各线快速入口

```bash
# 主线：Flask 评测台（零 key 可跑）
cd web-proto && python run_game.py             # 命令行单局
cd web-proto && python web_app.py              # Web 界面 :8860
cd web-proto && python train_rl.py             # 零成本 RL 训练（Mock 环境，不调 API）
cd web-proto && python eval_generalization.py  # held-out 泛化评测
cd web-proto && pytest tests/ -q               # 102/102

# 冻结线：MCP 服务器（归档，不再新增功能）
cd mcp-server && python run.py                 # 启动 MCP 服务器（stdio）
cd mcp-server && pytest                        # 45/45

# 工作区线（零 key 可跑）
cd workspace && python run/tournament.py --mock
```

## 共同约定

- API key 只放**各线自己的 `.env`**（`mcp-server/` 暂无 `.env`，复制 `.env.example` 自建）；`.env` 含真实 key，**不进归档、不外传、不进 git**
- 零 key 优先：mock 模式先跑通再优化；先度量后优化
- **跑实验（训练/评测/基准/长任务）一律在服务器执行，不在本机跑**：服务器 `172.16.222.99`（SSH user，ed25519 免密）；本机只做编码、静态检查、轻量单测；预计超过 10 分钟的任务一律上服务器。说明书：`E:/Program/实验服务器与全局配置.md`
- 对局日志即数据，勿删：`mcp-server/data/matches/`（68 局）、`web-proto/logs/`（6 局）、`workspace/logs/`（48 局）
- 每条线有自己的 `.workbuddy/memory/MEMORY.md` 进展记录，改动大版本后更新

## 安全须知

所有真实 API key 只存在于各线的 `.env`（已被根 `.gitignore` 排除），仓库内仅保留 `.env.example` 模板。克隆后复制模板、填入自己的 key 即可运行真实模型对战；缺 key 时 mock 模式仍可端到端跑通。

## 本机验证记录（2026-10-12 · Windows + Python 3.13，零 key 零成本）

| 线 | 验证项 | 结果 |
|---|---|---|
| `mcp-server/` | pytest 全套 | ✅ **45/45**（依赖固定 mcp 1.28.1 + fastmcp 3.4.4，提交 `bac7c06`） |
| `web-proto/` | `run_game.py` mock 端到端 / `test_llm_robust.py` / `test_connectivity.py` | ✅ 全过（容错 5/5；本机网络可达 zai/moonshot/deepseek/qwen，ashbwin 确认死亡） |
| `workspace/` | `run/tournament.py --mock` 32 局 + 聚合 / `pytest tests/` | ✅ 全过（mock 1.00 vs mock-weak 0.00 区分度正确；3/3） |

环境：`mcp-server` 用 venv `~/.workbuddy/binaries/python/envs/ai-arena`；两条原型线共用 `envs/ai-arena-proto`。安装走阿里云源（全局清华源曾取不到 setuptools）。真实模型对战涉及计费，待成本报批后再跑。

## 路线决策（2026-10-13 已定，不再是待定项）

| 决策 | 结论 | 理由 |
|---|---|---|
| **主线** | `web-proto/` 为唯一主线 | L1/L2 全部成果（含 102 例测试、指标口径、泛化协议）都在这条线上，继续投入边际成本最低 |
| `mcp-server/` | **冻结归档**，不删不弃 | v0.5 完整可用（45/45），其游戏机制已被主线吸收；保留作为可回溯资产，但不再新增功能 |
| `workspace/` | 保持资产库定位 | 文档与编排脚本，按需取用 |
| 新工作落点 | 一律在 `web-proto/` | 避免双线维护成本（此前两线并行是明确的待解问题） |

> 历史背景：MCP vs Flask 的路线之争在各线 MEMORY.md 中长期记为「未定」。
> 本次拍板的实质不是「哪个架构更好」，而是**已有成果集中在哪条线上**——
> 选择依据是资产分布而非架构偏好。

## 整理记录（2026-10-12）

- 三个历史压缩包（v0.5-full.zip / full.tar.gz / backup_20261012_0342.zip）已解压并入本结构，与包内内容**字节级 diff 校验一致**后删除
- 已清除垃圾：`__pycache__` × 4、`.pytest_cache`、空 `.tmp`
- 备注：`mcp-server/` git 工作区缺一个被跟踪的 `AI-Arena-src.zip`（v0.5 打包时把自身归档提交进了仓库，属固有自引用），不影响源码与测试
