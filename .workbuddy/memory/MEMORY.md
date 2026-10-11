# AI 角斗场 · 进展记忆 (MEMORY)

> 自动维护：每次大版本更新时记录。工作目录 `AI-Arena/`（沙箱内为 `/workspace`）。

## 项目定位
AI vs AI 社交推理评测系统：让不同 LLM 在海龟汤/狼人杀里对抗，评估推理/欺骗/伪装/辩论能力。
本质是离线评测系统（非真人游戏）。技术栈：Python 3.13（沙箱 3.11）、litellm 统一多模型、JSON 日志 + pandas/matplotlib。
核心约束：代码/日志只放 AI-Arena 内；敏感 key 进 `.env`；先跑通再优化；报告要可读（图表+文字，便于发小红书/比赛）。

## v0.1（已完成 · 2025-07-13）
**交付：可运行 MCP 服务器脚手架（stdio 传输 + Mock 玩家先跑通，无需 API key）**

架构：
- `mcp_server/server.py` — FastMCP 入口，注册 tools/resources/prompts
- `mcp_server/config.py` — 读 `.env`、模型注册表（gpt/deepseek/qwen/kimi 等）
- `mcp_server/engine/` — `puzzles`(题库) `game`(引擎) `player`(MockPlayer+LLMPlayer) `judge`(规则裁判) `prompts`(提示词)
- `mcp_server/eval/` — `metrics`(基础指标+排行榜) `report`(pandas/matplotlib 报告，已注册中文花字体回退)
- `mcp_server/store/logs.py` — JSON 对局日志落盘 `data/matches/`
- `mcp_server/data/puzzles.json` — 4 个海龟汤示例题
- `tests/` — 引擎测试 + MCP 内存客户端端到端测试（pytest 全绿 7/7）

MCP 接口：10 个 tools（list_puzzles/get_puzzle/start_match/get_match_state/step_match/run_match/list_matches/analyze_match/get_leaderboard/export_report）、3 个 resources（arena://puzzles、arena://matches/{id}、arena://models）、3 个 prompts（player_turn/judge_answer/debrief）。
安全：`get_puzzle` 默认不泄露 truth；玩家禁止以 include_truth=true 调用。

验证结论：
- 引擎：Mock 玩家可端到端跑通到 finished，含猜中即胜与回合上限兜底。
- MCP 传输：stdio 入口 `python run.py` 实测可用（列工具/取题/跑局正常）；FastMCP 内存客户端端到端全绿。
- 报告：leaderboard.png 中文渲染正常（已注册 Noto CJK / 回退英文标签）。

踩坑记录：
- FastMCP 3.x `call_tool` 返回 `Output` 对象，dict 返回在 `.data`、list 返回在 `.data.result`；资源返回 content 列表；prompt 返回 `GetPromptResult.messages[].content.text`。测试解析需分别处理。
- pandas `DataFrame.empty` 是属性不是方法（report.py 曾误当可调用）。
- matplotlib 中文需主动注册 CJK 字体，否则乱码/警告。

## v0.2（进行中 · 2026-10-11）
**主题：让评测真正成立 —— 启发式玩家 + 题库扩充**

### 为什么要做
v0.1 的 MockPlayer 是掌握真相的"预言机"，两个 mock 对战必定打平且都满分，
排行榜毫无意义。评测系统要成立，必须有**不作弊、且彼此存在水平差异**的玩家。

### 已交付
- **HeuristicPlayer**（`engine/player.py`）：只持有汤面(surface)，不接触 truth/key_facts
  （已加 `uses_truth=False` 自检标记与单元测试）。靠「汤面 2 字词 + 常识假设库」提问，
  用裁判 yes/no 累积线索后重构猜想。四种性格在**三个维度**分化：

  | 性格 | 提问顺序 | 收网时机 | 猜想用几条线索 |
  |---|---|---|---|
  | aggressive 激进 | 通用假设优先 | 30% | 2 条（草率） |
  | conservative 保守 | 汤面细节优先 | 85% | 全部（扎实） |
  | balanced 均衡 | 交替 | 60% | 5 条 |
  | creative 跳跃 | 完全打乱 | 55% | 全部 |

- **题库 4 → 12 题**（`data/puzzles.json`），覆盖 easy/medium/hard 与悬疑/推理/反转/
  温情/密室/法律等标签。写题时刻意让 key_facts 保留汤面实词，使启发式提问能真实命中。
- **裁判判定重写**（`engine/judge.py`）：Jaccard → 覆盖率 + 连续 2 字命中。

### 排行榜实测（12 题 × 4 性格）
| 玩家 | 胜率 | 平均命中率 |
|---|---|---|
| conservative | 33% | 0.208 |
| aggressive | 33% | 0.181 |
| creative | 17% | **0.264** |
| balanced | 8% | 0.250 |

结论：**命中率 ≠ 胜率**。规则是"先猜中即胜"，早收网能吃先手红利；
creative 命中率最高却赢不多，典型的"看得准但慢一步"。

### 踩坑
- 分词用逐字滑动窗口会产生「男人每 / 人每天」这类跨词噪声，改成
  **2 字不重叠切分**（步长 2）后候选词质量明显变好。
- 纯 Jaccard 判定会被提问模板词稀释（问题越长重叠度越低，长事实几乎不可能命中）。
  改为「覆盖率 ≥ 0.15 **或** 含连续 2 字实词」后命中率正常；
  LCS 必须排除纯虚词（他是/的了），否则任何带"他是"的提问都能命中。
- 只让性格差在"提问数量"上，线索集合是子集关系、得分会打平；
  必须让**提问顺序 + 收网时机 + 猜想完整度**三个维度都分化，区分度才稳定。

## v0.3（进行中 · 2026-10-11）
**主题：评测公平性 + 多维指标 + 雷达图**

### 1. 修掉先手优势：轮末统一结算
原逻辑是玩家猜中瞬间 `_finish(winner)`，后手玩家**连提交猜想的机会都没有**——
评测到的其实是"手速 + 出场顺序"而非推理能力。实测激进派凭此拿到最高胜率。

改法：`step()` 猜中只记入 `pending_guesses`，等一轮内所有玩家都行动完，
由 `_settle_round()` 统一比较，命中率最高且达标者胜。

效果：保守派胜率 33% → **42% 登顶**；creative 平均命中率 0.264 → **0.333**
（后手终于有完整一轮时间作答）。

### 2. 四维推理画像（供雷达图）
| 维度 | 含义 | 激进 | 均衡 | 保守 | 跳跃 |
|---|---|---|---|---|---|
| accuracy | 猜想命中率（说得对不对） | 0.181 | 0.250 | 0.208 | 0.250 |
| efficiency | 每次提问命中多少事实（问得准不准） | **0.219** | 0.153 | 0.146 | 0.114 |
| breadth | 触及事实占全题比例（挖得广不广） | 0.292 | 0.347 | **0.361** | 0.264 |
| decisiveness | 多早就敢提交猜想（敢不敢下结论） | **0.625** | 0.368 | 0.285 | 0.378 |

读图：激进派"最快也最准地问，但结论下得草率"；保守派"挖得最广却最不敢收网"。

### 3. 作弊玩家隔离
`CHEATING_PREFIXES=("mock",)`，`compute_leaderboard/profile` 默认
`exclude_cheating=True`；MCP `get_leaderboard(include_cheating=False)` 同步默认剔除。

### 4. 雷达图
`report.generate_radar()` → `reports/radar.png`，极坐标 + 中文标签渲染正常。

> ⚠️ **诚实边界**：项目目标是推理/欺骗/伪装/辩论四维能力，但后三者需对抗模式
> （知情者可撒谎、辩论玩法）才能观测。当前雷达图只反映**推理侧四种风格**，
> 不是完整四维能力评分。评测报告里已明确标注这一点。

### 踩坑
- **作弊玩家不能混排**：把 mock 放进启发式玩家对战，它会立刻碾压终结对局，
  导致其他人 accuracy/decisiveness 全为 0。mock 只能单独跑或用于流程验证。
- 改 `compute_leaderboard` 默认 `exclude_cheating=True` 会让旧测试失败
  （旧断言期望含 mock），必须同步更新测试。

## v0.4（进行中 · 2026-10-11）
**主题：对抗模式 —— deception 欺骗玩法（补齐"欺骗/伪装"维度）**

### 已交付
- **deception 模式**：`start_match(..., mode="deception")`，**首位玩家是知情者(insider)**，
  掌握真相并可撒谎；其余是求解者。回合只在求解者间轮转，知情者只负责应答。
- **InsiderPlayer** 三种策略：`honest`(0% 撒谎，对照组) / `misleading`(100%) / `mixed`(50%)。
- **欺骗指标**：`deception_rate`、`avg_solver_accuracy`（`deception_metrics()`）。

### 实测（12 题 × 保守 + 激进两种求解者）
| 知情者策略 | 撒谎率 | 求解者平均命中率 |
|---|---|---|
| honest 诚实（对照） | 0% | 0.243 ← 基准线 |
| mixed 摇摆 | 47% | 0.104 |
| misleading 全骗 | 100% | **0.000** |

结论：**撒谎率与求解者命中率强负相关**，欺骗能力首次变成可量化指标。
反直觉的一点：`misleading` 全程撒谎反而"规律过于明显"（求解者只要反着听即可），
实战中 `mixed` 真话假话混着说才最难对付——后续值得加"识破规律"的求解者策略来验证。

### 安全设计
`_lie_log`（撒谎记录）**落盘保留**供离线评测，但**不进玩家可见状态**
（`get_state` 走 `_public_state` 已剔除）。否则求解者直接查日志就能看穿谎言。

### 踩坑
- 把过滤条件从 `is_cheating` 换成 `is_rankable` 时**忘了去掉取反 `~`**，
  结果排行榜把求解者排除、只留知情者。语义反转的过滤器必须同步改布尔方向。
- 知情者"知道答案"是**角色设定**（正如真人海龟汤的汤主本就该知道），不是作弊，
  必须和 mock 严格区分：mock 被排除是因为成绩不可信，insider 被排除是因为他根本不是求解者。

### challenge 质疑动作（伪装维度补完）
**测谎原理**：求解者按 `repeat_rate` 概率重复提问同一个问题，
若前后回答不一致即抓到谎言——这是真人玩海龟汤的常用手法。

复核倾向 `repeat_rate`：保守 0.30 / 均衡 0.15 / 跳跃 0.10 / 激进 0.05。

核验方式：玩家 challenge 时带 `target`（具体哪个问题前后矛盾），
引擎查 `_lie_log` 中该问题是否存在撒谎记录，有则质疑成立。

**实测（ts-001，8 局平均）**

| 知情者 | 撒谎率 | 保守派质疑次数 | 激进派 | 准确率 |
|---|---|---|---|---|
| honest | 0% | 0 | 0 | — |
| mixed | 57% | **1.9** | 0.1 | 100% |
| misleading | 100% | **0** | 0 | — |

**伪装指标（12 题平均）**

| 知情者 | 撒谎率 | 被抓率 | 未识破谎言 | 伪装分 | 求解者命中率 |
|---|---|---|---|---|---|
| honest | 0% | 0% | 0.0 | N/A | 0.306 |
| mixed | 50% | 29% | 4.8 | 0.71 | 0.222 |
| misleading | 100% | 0% | 14.8 | 1.00 | **0.000** |

三条洞察：
1. **复核倾向真实转化为识破能力**：保守派 1.9 次 vs 激进派 0.1 次，差 19 倍。
2. **misleading 抓不到（0 次）**——它稳定说反话，重复问答案完全一致。
   这是求解者能力的真实局限，不是 bug：破解它需要"反着听"的策略，当前未实现。
3. **伪装分高 ≠ 欺骗成功**：misleading 伪装分满分，是因为当前求解者没能力抓它；
   它真正强在把求解者命中率打到 0.000。两个指标必须分开看。

### 踩坑
- 质疑核验最初查"最近一条回答"，与玩家指出的**具体矛盾问题**对不上，导致误判失败。
  改为按 `target` 问题核验 `_lie_log` 才正确。
- 一次谎都没撒时 `disguise_score` 显示 1.00（满分），会被误读成"伪装大师"。
  改为 `None`（不适用）——诚实者没有伪装可言。

### 仍待补齐
- ~~**辩论维度**~~ ✅ v0.5 已补齐（debate 模式 + 比较制 + 辩论指标）。
- **破解 misleading**：需要"反向利用规律"的求解者策略（识别稳定反向回答并取反）。

## v0.5（已完成 · 2026-10-12）
**主题：辩论模式（补齐第四维）+ 平局公平性修复**

### 已交付
- **debate 模式**：`start_match(..., mode="debate")`。收网动作从 `guess` 变为 `argue`
  （提交论证而非单纯猜想）。**比较制**：不看绝对阈值，相对对手更强即胜——
  这是辩论对抗性唯一的观测入口。
- **辩论指标**（`debate_metrics()`）：`argument_accuracy` 论证质量、
  `persuasion` 说服力（己方命中/全场最佳）、`evidence_used` 论据充分度、`ranking`。
- **平局加时赛**（`_break_tie()`）：分数并列时不再按出手顺序送胜——
  - debate 比**论据充分度**（各自 yes 提问数）；
  - qa 比**提问效率**（同样命中率下提问更少者胜）；
  - 副标准仍持平 → `winner=None`（无人胜出），宁可比赛继续也不按字典序送胜。

### 实测（4 性格两两对战 × 4 题，24 场）
激进 9 胜 / 跳跃 9 胜 / 均衡 5 胜 / 保守 1 胜，**0 平局**。
洞察：覆盖率制评分下**话多反而摊薄分数**——堆砌全部线索不如精简命中，
保守派"广撒网"风格在辩论里最吃亏。

### 踩坑（两个都值钱）
1. **风格名只认连字符不认冒号**：`heuristic:aggressive` 被静默 fallback 成
   balanced，两个"不同性格"玩家行为逐字相同，18 场 debate 全部平局、区分度为零。
   修法：`lstrip(":-_ ")` 同时支持两种分隔符，并补回归测试。
   **教训：静默 fallback 比报错更危险**——它让"看起来在跑"的系统产出全平局，
   而全平局又会被误读成"双方真的势均力敌"。
2. **上传参数 file_size 误传"包内文件总大小"**（393038）而非 zip 实际字节数
   （260967）。PUT 的是实际文件流所以内容无损（MD5 校验通过），但预分配的
   元数据是错的。`unzip -l` 末尾数字是**内容总大小**，不是文件大小，
   `stat -c %s` 才是。

### v0.5 收尾状态
- pytest **45/45** 全绿（新增 tests/test_debate.py 11 项）。
- git commit `be0e62e`；资产盘 `AI-Arena-src.zip` = `UjyMmsXWcJOY`
  （260967 字节，MD5 `9bfa17d7f3a0a35e30387c3a6d2d87ec`，下载回校验一致）。

## 待办 / 路线图
- v0.2（部分完成）：✅ 启发式玩家 + 题库 12 题；⏳ 真实多模型对战（litellm，需 .env key）、LLM 裁判
- v0.3：✅ 轮末统一结算、四维画像、雷达图、作弊隔离
- v0.4：✅ deception 欺骗模式、知情者三策略、challenge 质疑动作、欺骗/伪装指标
- v0.5（当前）：✅ debate 辩论模式、比较制结算、平局加时赛、辩论指标、风格名解析修复
- v0.6（可选）：破解 misleading 的"反向利用"求解者策略、HTTP/SSE 传输、小红书风格报告导出

## 环境备注
- 沙箱 Python 3.11，`pip install -e .` 装 `mcp` 1.28.1 + pandas/matplotlib；`fastmcp` 3.4.4 作为测试客户端（dev 依赖）。
- `tests/test_server.py` 用 `fastmcp.Client(srv.mcp)` 内存模式验证 MCP 层，规避 stdio 手动解析的版本差异。

## 落盘 / 分发记录（2026-10-12 · v0.5 更新）

**资产盘（tdrive）**：root `DKobgCurEQbo` → 项目目录 `AI-Arena` = `DoXEjAujQUMq`。

| 通道 | 状态 | 位置 |
|---|---|---|
| 沙箱 `/workspace` | ✅ 主工作副本 | 52 个文件（源码/题库/测试/报告/样例对局日志/MEMORY.md） |
| 资产盘 tdrive | ✅ 已落盘 | `AI-Arena-src.zip` = `UjyMmsXWcJOY`（v0.5：262950 字节，MD5 `9107170fe1787085cce7f80a126d676c`，下载回本地校验逐字节一致）。同一 file_id 多次覆盖更新，MD5 随之变化，校验时以资产盘实际下载为准。注：zip 包内的 MEMORY/DEPLOYMENT 文档为打包时快照（MD5 `9bfa17d7...`），核心代码与本地逐字节一致 |
| git 本地 | ✅ 已提交 | v0.5 = commit `be0e62e`（v0.1 = `007bfe7`）；**无 remote，未推送** |
| 乐享知识库 | ✅ 已发布（v0.1 内容） | 个人空间 `459b0f14579d4ba98001b5aef3a7d595`；`AI-Arena` 文件夹 `96e71c854a2d4f80964e3c179350f94c`；README 页 `79c3b93a4298478f8064cef6132dd169`、架构设计页 `7271888faec348efa3e35cf94ae89f68` |

资产盘现状：压缩包外的散落文件仅 `README.md`（`UXqQriRHhOhf`）和 `engine/__init__.py`（`UrCanHujNRAL`），其余目录（`tests/`、`eval/`、`store/`、`reports/`、`.workbuddy/memory/`、`data/`）均为**空壳**，完整内容以 zip 为准。

踩坑：
- 网盘上传是三步：`file_upload` 取预签名 → `curl -T` PUT → `file_upload_complete`。只做第一步**不会落库**，上一轮"9 个文件已获凭证"实际全是空目录。
- **PUT 必须带 `x-cos-security-token` 头**（预签名返回的 headers 里）：漏带会报 `InvalidAccessKeyId`，COS 临时 AK 不配安全令牌就"不存在"。
- 网盘拒绝点开头文件名（`.gitignore`/`.env.example` 会被判为"文件名为空"），进压缩包可规避。
- **打包 zip 一次上传是海量小文件的最优解**：把 75+ 次顺序调用压缩到 3 次，且目录结构完整保留。
- `dir_list` 对刚上传文件可能返回 `size=0`（元数据延迟），必须用 `file_download` + MD5 对比才能真正验证完整性，不能只看文件名。
- `file_upload` 的 `file_size` 要用 `stat -c %s` 的**文件实际字节数**，不是 `unzip -l` 末尾的"内容总大小"（v0.5 曾误传 393038 vs 实际 260967；PUT 文件流无损，MD5 校验仍通过）。
- 网盘文件名与扩展名分离存储（name=`AI-Arena-src`，ext=`zip`）。
