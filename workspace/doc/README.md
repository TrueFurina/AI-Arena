# AI-Arena 项目文档

> AI vs AI 社交推理游戏评测台 · 文档中心

## 项目一句话定义

AI-Arena 是一个**离线 AI 对战评测台**：让不同的 LLM 在社交推理游戏（海龟汤起步，后续狼人杀）里当玩家相互对抗，把对局过程结构化记录成 JSON 日志，再用量化指标横向比较各模型在 **推理 / 欺骗 / 伪装 / 辩论** 四个维度上的能力差异。

我们造的不是游戏产品，而是「能力探针 + 评测报告生成器」。最终交付物：可复跑的对局数据 + 可读的评测报告（图表 + 文字结论）。

## 边界（重要）

- ❌ 不做真人联机（那是已挂起的 Claw 剧本杀项目，方向不同，勿混淆）
- ✅ 所有对局是 AI vs AI 离线跑批
- ✅ 所有代码 / 文件 / 日志都在 AI-Arena 目录下
- ✅ API key 等敏感配置统一放 `.env`，不写死在代码里
- ✅ 评测报告要可读（图表 + 文字结论），方便发小红书 / 比赛

## 技术选型

- **Python 3.13**（LLM SDK 生态好、数据分析方便）
- **litellm** 统一调多家模型（OpenAI / Anthropic / DeepSeek / 通义 / Kimi 等）
- **JSON 对局日志 + pandas / matplotlib** 分析

## 文档导航

| 文档 | 内容 |
|---|---|
| `DESIGN.md` | 深度规划设计：项目本质、玩法设计、架构、适配层、评测、里程碑、验收 |
| `ARCHITECTURE.md` | 技术架构与目录结构、模块职责、数据流 |
| `GAME.md` | 海龟汤游戏机制、角色定义与变体设计（A/B/C/D） |
| `EVALUATION.md` | 四维度评测体系与量化指标定义 |
| `ROADMAP.md` | 落地里程碑路线 M0–M6 |

## 当前状态

> ⚠️ 项目处于**零文件基线**：目录已建，但代码、题库、`.env`、`MEMORY.md` 均未创建。下一阶段从 `ROADMAP.md` 的 **M0** 起步。

## 快速开始（规划中，随 M0–M4 落地生效）

```bash
# 1. 安装依赖
pip install -r requirements.txt
# 2. 配置 key（复制模板，填入真实 key，不要写死在代码）
cp .env.example .env
# 3. 跑通一局（mock 模式，无需任何 key）
python run/single_game.py --mode mock
# 4. 生成评测报告
python eval/report.py
```
