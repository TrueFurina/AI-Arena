# 落地里程碑路线 M0–M6

## 里程碑总表

| 里程碑 | 内容 | 交付物 | 依赖 key | 状态 |
|---|---|---|---|---|
| **M0 脚手架** | 建目录结构 + mock_player 跑通一局 | 端到端可运行 demo（假数据） | ❌ | ✅ |
| **M1 单模型单局** | 真实 LLM 当提问者，GM 用规则/脚本 | 单局 JSON 日志 | ✅(真实调用) | 🟡 代码就绪，key 阻塞 |
| **M2 题库 + 指标 v1** | 扩充题库、实现推理指标 | 单局评测分 | ❌(M1 已有代码) | ✅ |
| **M3 多模型对战** | tournament 循环、模型注册表 | 多模型对比数据 | ❌(mock 可跑) | ✅ 零 key 验证通过 |
| **M4 评测报告** | 可读战报（小红书/比赛友好） | `SHARE.md` + `report_share.py` | ❌ | ✅ |
| **M5 骗子变体** | 欺骗维度：GM 撒谎 + 鲁棒性 | DeceiverGM + `deceiver_demo.py` + spec | ❌(PoC) | 🟡 PoC 落地 |
| **M6 辩论环节** | 辩论 + LLM-as-Judge | `debate.py` + `debate_demo.py` + spec | ❌(PoC) | 🟡 PoC 落地 |

> 起步任务四条（引擎+题库 / 适配层 / 单模型单局 / 多模型+评测）≈ **M0–M4**，是现在就要做的核心。
> M5/M6 已先以**零 key PoC** 验证引擎层通畅，真实模型接入点用 `completion_fn` 依赖注入预留。

## 当前进度（更新于 2026-07-17）

- [x] **M0 脚手架** ✅ 完成：引擎 + mock 玩家 + 单局编排，零 key 跑通
- [x] **M1 单模型单局** ⚠️ 代码完成，但**真实调用被 key 阻塞**（见下「API 现状」）。dry-run/冒烟/零 key 链路全绿
  - 🛠 **零 key 保障改进**：`litellm` 改为**惰性导入**（`players/llm_player.py`），`--mock` 路径与测试不再因未装 litellm 崩溃，真正落实「零 key 优先」
- [x] **M2 题库 + 指标 v1** ✅ 完成：题库 8 题 + SmartGM 关键词感知 + `eval/metrics.py` + `run/evaluate.py`（还原度/汤底保真/效率）
- [x] **M3 多模型对战（脚手架，零 key 可跑）** ✅ `config.toml` 模型注册表 + `run/tournament.py` 批量对战 + 自动聚合；含 mock-strong/mock-weak 零 key 基线（实测区分度 1.0 vs 0.0）；真实模型待填 key 启用
- [x] **M4 评测报告** ✅ `run/report_share.py` 读 `metrics.csv` → 生成 `analysis/SHARE.md`（战力榜 + 各题矩阵 + 指标速读 + 小红书/比赛文案提示）+ 刷新 `restoration.png`；配 `arena-report` skill
- [x] **M5 骗子变体（PoC）** 🟡 `players/gm.py::DeceiverGM`（按比例翻转 yes/no，记录 `lied`）+ `run/deceiver_demo.py`（同提问者跨诚实/撒谎环境算 `deception_robustness`）；`eval/metrics.py` 增 `lie_rate`/`deception_robustness`；配 `arena-deceiver` skill 与 `specs/deceiver-design.md`
- [x] **M6 辩论 + Judge（PoC）** 🟡 `engine/debate.py`（`Debater`/`MockDebater`/`Judge`/`MockJudge`/`Debate` 状态机）+ `run/debate_demo.py`（正反多轮对抗 + 结构化 verdict）；真实 LLM 接入点用 `completion_fn` 预留；配 `arena-debate` skill 与 `specs/debate-design.md`

## API 现状与阻塞（关键）

> 真实 LLM 对局当前**被 key 阻塞**。已实测确认：

1. **tokenc.top 不能当模型后端**。用户提供的 `ag_...` 令牌有效，但该平台是「通用工具 API 聚合网关」（翻译/天气等）。实测：
   - `translate` 服务 `success:true`（网关本身通）；
   - `chat` / `llm` / `openai` / `ai` / `models` 服务全部 `404 PROVIDER_NOT_FOUND`（「暂未接入」）。
   - 结论：它**不提供 LLM 对话能力**，无法用于 AI-vs-AI 对局。
2. **netdrive MCP 本会话故障**：读 `API_KEYS.md` 附件时 `file_download`/`dir_list` 连续报 `Expected object, received string`（环境层参数解析异常），附件也不在 `/root/uploads`。真实 LLM key 暂时取不到。
3. **代码侧已 100% 就绪、零 key 可验证**：
   - 测试全绿（3 passed）；`--dry-run` 验证提示词/GM/题库/别名解析正常；
   - litellm 自动按 `DEEPSEEK_API_KEY`/`DASHSCOPE_API_KEY`/`MOONSHOT_API_KEY`/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` 读取；
   - 自定义 OpenAI 兼容端点只需 `OPENAI_API_BASE`+`OPENAI_API_KEY` + `--model openai/<名>`；
   - 新增 `run/test_api.py`：一键自检所有供应商连通性 + tokenc 网关探测（零 key 优雅 SKIP）。

**解锁条件**：用户提供一个真实 LLM key（DeepSeek/Qwen/Kimi/OpenAI/Claude 任一，贴到对话即可）→ 写入 `.env` → 一条命令跑真实对局。

## 下一步规划（M3 起）

遵循「先跑通再优化」：**M3 的脚手架现在就能零 key 搭好**，key 到位即一键跑真实对战。

### 可立即做（零 key，不阻塞）
- [ ] **M3 脚手架**：
  - `config.toml` 模型注册表：每项 `{alias, litellm_model, enabled, note}`，支持从注册表批量跑；
  - `run/tournament.py`：对注册表中每个 `enabled` 模型 × 每题跑 N 局（vs SmartGM），日志落 `logs/`；
  - 零 key 时用 mock 玩家验证编排正确性，真实模型靠 `.env` 自动切换。
- [ ] **M4 报告增强**：多模型横向对比图（还原度/效率/汤底保真分组柱状图）+ 文字结论模板，适配小红书/比赛可读性。
- [ ] 题库继续扩（目标 20+ 题），补充 `keywords`/`clue_keywords` 梯度。

### 依赖 key（拿到真实 LLM key 后）
- [ ] M3 真实多模型对战，产出 `analysis/metrics.csv` 跨模型聚合；
- [ ] M5 骗子变体（诚实者/说谎者角色），评测**欺骗/伪装**维度；
- [ ] M6 辩论环节 + LLM-as-Judge，评测**辩论**维度；至此四维度（推理/欺骗/伪装/辩论）全覆盖。

## 验收标准（做到什么算「完成」）
1. 一条命令跑通一局，输出结构化 JSON 日志到 `logs/`；
2. 一条命令生成评测报告（含图表 + 文字结论）到 `analysis/`；
3. 多模型可在 `config.toml` 里配置并自动循环对战；
4. 全程零 key 也能用 `mock_player` 跑通 M0（保证 pipeline 不卡在 key 上）。

## 建议下一步动作（待你确认）
- 选项 A（推荐）：你贴一个真实 LLM key → 我立刻跑通 M1 真实单局 + 顺手搭 M3 脚手架；
- 选项 B：先不拿 key，我直接零 key 把 M3 脚手架（config.toml + tournament.py）搭好，key 到位即跑；
- 选项 C：你重新提供能用的 LLM 网关（base_url + key），我按自定义端点接入。

## 文档与技能清单（本项目已沉淀）

> 详见各文件。核心方法论见 `doc/METHODOLOGY.md`。

### 文档（doc/）
| 文件 | 内容 |
|---|---|
| `VISION.md` | **项目愿景与四维度蓝图（北极星）**：定位、设计哲学、A推理/B双盲互猜/C骗子变体/D辩论 演进、M0–M6 总览、社区/比赛定位 |
| `METHODOLOGY.md` | **编程方法论（核心）**：零 key 优先、结构化日志即可观测性、先度量后优化、toil 自动化、渐进式接入、代码组织、测试策略、反模式 |
| `ROADMAP.md` | 里程碑 M0–M6 + 当前进度 + API 现状与阻塞 + 下一步规划 |
| `EXTENDING.md` | 扩展指南：加模型 / 加题库题目 / 加游戏模式（A–D） |
| `KEYS.md` | 密钥与 API 连接：`.env` 管理、供应商对照、`test_api.py` 自检、自定义端点、tokenc 教训 |
| `EVALUATION.md` | 评测指标定义（restoration/guess_score/efficiency）+ 四维度蓝图 + 报告阅读 |
| `QUICKSTART.md` | 5 分钟上手（venv / 零 key mock / dry-run / 贴 key 跑真实 / 出报告 / 测试） |
| `TROUBLESHOOTING.md` | 常见坑（无 key 假成功、tokenc 非 LLM、中文方框、netdrive 故障、PEP668、旧日志混入等） |
| `ARCHITECTURE.md` / `DESIGN.md` / `GAME.md` / `README.md` | 早期规划文档（架构/设计/玩法/概览） |

### 技能（.codebuddy/skills/，可用 Skill 工具一键调用）
| skill | 用途 |
|---|---|
| `arena-run` | 运行单局/对局（`single_game.py`，mock/llm/dry-run） |
| `arena-evaluate` | 聚合日志生成报告（`evaluate.py` + `metrics.py`） |
| `arena-puzzle` | 新增/校验题库题目（字段与 keywords 格式） |
| `arena-metrics` | 解释与复算推理指标（restoration/guess_score/efficiency） |
| `arena-tournament` | M3 多模型对战（`tournament.py` + `config.toml` 注册表） |
| `arena-report` | M4 可读战报（`report_share.py` → `SHARE.md`，小红书/比赛友好） |
| `arena-deceiver` | M5 骗子变体（`deceiver_demo.py` 演示，deception_robustness） |
| `arena-debate` | M6 辩论 + Judge（`debate_demo.py` 演示，persuasion_margin） |

### 规格与记忆
- `specs/engine-design.md`：引擎状态机 + JSON 动作协议详细规格（§5/§7 等）
- `specs/tournament-design.md`：M3 多模型对战设计（注册表 schema / 循环 / 输出闭环 / 零 key 验证）
- `specs/deceiver-design.md`：M5 骗子变体设计（V1/V2/V3 变体、协议扩展、指标、反模式）
- `specs/debate-design.md`：M6 辩论 + LLM-as-Judge 设计（架构、verdict 协议、指标、真实接入）
- `.workbuddy/memory/MEMORY.md`：按规则 3 持续记录进展与关键决策
