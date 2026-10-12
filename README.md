# AI-Arena · 项目总览

AI vs AI 社交推理游戏评测台（海龟汤起步）：让不同 LLM 在**推理 / 欺骗 / 伪装 / 辩论**四个维度上对抗，对局过程落结构化 JSON 日志，用量化指标横向比较各模型能力。本质是「能力探针 + 评测报告生成器」，不是游戏产品。

## 目录结构（三条开发线）

| 目录 | 路线 | 状态 |
|---|---|---|
| `mcp-server/` | MCP 服务器（FastMCP / stdio） | ✅ **v0.5 已完成**（2026-10-12）：启发式玩家四性格、12 题题库、轮末统一结算、deception 欺骗模式 + challenge 质疑、debate 辩论模式（比较制 + 平局加时赛）；pytest 45/45；git 4 个提交（含依赖界修复 `bac7c06`）；已部署资产盘（MD5 校验一致） |
| `web-proto/` | Flask 网页原型（端口 8860） | 🟡 原型可跑：`zai/glm-4-flash` 真实模型端到端已验证；llm.py 加固版（重试 + 角色降级）；ashbwin 免费端点不可达（可弃） |
| `workspace/` | 文档 / 技能 / M0–M6 脚手架线 | 📚 设计资产库：12 篇设计文档（`doc/`）+ 4 篇规格（`specs/`）+ 8 个项目技能（`.codebuddy/skills/`）+ tournament/evaluate/report_share 等编排脚本；游戏机制概念已被 `mcp-server/` 吸收实现 |

## 各线快速入口

```bash
# MCP 主线（零 key 可跑）
cd mcp-server && python run.py                 # 启动 MCP 服务器（stdio）
cd mcp-server && pytest                        # 45/45

# Flask 原型（零 key 可跑）
cd web-proto && python run_game.py             # 命令行单局
cd web-proto && python web_app.py              # Web 界面 :8860

# 工作区线（零 key 可跑）
cd workspace && python run/tournament.py --mock
```

## 共同约定

- API key 只放**各线自己的 `.env`**（`mcp-server/` 暂无 `.env`，复制 `.env.example` 自建）；`.env` 含真实 key，**不进归档、不外传、不进 git**
- 零 key 优先：mock 模式先跑通再优化；先度量后优化
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

## 路线决策（待用户定夺）

MCP 服务器 vs Flask 网页版的最终路线在各线 MEMORY.md 中均记录为**未定**。当前以 `mcp-server/` 为最新完成主线。

## 整理记录（2026-10-12）

- 三个历史压缩包（v0.5-full.zip / full.tar.gz / backup_20261012_0342.zip）已解压并入本结构，与包内内容**字节级 diff 校验一致**后删除
- 已清除垃圾：`__pycache__` × 4、`.pytest_cache`、空 `.tmp`
- 备注：`mcp-server/` git 工作区缺一个被跟踪的 `AI-Arena-src.zip`（v0.5 打包时把自身归档提交进了仓库，属固有自引用），不影响源码与测试
