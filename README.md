# AI 角斗场 (Arena) · MCP 服务器 (v0.5)

让不同的 AI 模型在海龟汤（社交推理游戏）里相互对抗，通过对局数据评估各模型的
**推理 / 欺骗 / 伪装 / 辩论** 能力。本仓库是评测系统的 **MCP 服务器**，把游戏引擎 +
多模型对战 + 自动评测封装成标准 MCP 工具，供任意支持 MCP 的客户端调用。

> 本质：AI vs AI 的离线评测系统，不是真人游戏产品。
> v0.4 起内置 **启发式玩家**与 **欺骗对抗模式**，**无需任何 API key** 即可跑出
> 有区分度的排行榜、能力雷达图与欺骗指标；填入 `.env` 后可直接换成真实大模型对战。
> v0.5 补齐**辩论模式**，推理/欺骗/伪装/辩论四维全部可观测。

## 快速开始

```bash
pip install -e .            # 安装本体（依赖 mcp / pydantic / pandas / matplotlib）
python run.py               # 以 stdio 方式启动 MCP 服务器
```

不接 key 也能跑出有意义的评测：

```python
from mcp_server import server as srv

# ① 问答模式：四种性格的启发式玩家同台竞技
players = ["heuristic-aggressive", "heuristic-conservative",
           "heuristic-balanced", "heuristic-creative"]
mid = srv.start_match("ts-001", players, max_rounds=24)["match_id"]
srv.run_match(mid)
srv.get_leaderboard()       # 排名（默认剔除作弊玩家与知情者）
srv.get_profiles()          # 四维推理画像
srv.export_report()         # 排行榜图 + 雷达图 + 文字报告

# ② 欺骗模式：知情者可撒谎，考验求解者识破能力
mid2 = srv.start_match(
    "ts-001",
    ["insider-mixed", "heuristic-conservative", "heuristic-aggressive"],
    mode="deception", max_rounds=20,
)["match_id"]
srv.run_match(mid2)
srv.analyze_match(mid2)     # 含 deception 欺骗指标

# ③ 辩论模式：比较制，谁的论证更有说服力谁赢
mid3 = srv.start_match(
    "ts-001",
    ["heuristic-conservative", "heuristic-aggressive"],
    mode="debate", max_rounds=16,
)["match_id"]
srv.run_match(mid3)
srv.analyze_match(mid3)     # 含 debate 辩论指标（说服力/论据充分度）
```

## 玩家类型

| 玩家名 | 说明 | 能否参与求解者排名 |
|---|---|---|
| `heuristic` / `heuristic-{aggressive,conservative,creative}` | **启发式求解者**：只拿到汤面，靠「汤面词 + 常识假设」提问，用裁判反馈累积线索后重构猜想 | ✅ 主力 |
| `insider` / `insider-{honest,misleading,mixed}` | **知情者**（欺骗模式）：掌握真相并可撒谎误导，评估的是欺骗/伪装能力 | ❌ 不是求解者 |
| `mock` | 预言机：掌握真相，用于验证引擎流程是否跑得通 | ❌ 作弊，成绩不可信 |
| `gpt-4o-mini` / `deepseek-chat` / `qwen-plus` / `kimi` … | 真实大模型（经 litellm 调用，需在 `.env` 填 key） | ✅ |

四种求解者性格在**三个维度**分化，因此得分天然不同（这是排行榜有意义的前提）：

| 性格 | 提问顺序 | 收网时机 | 猜想用几条线索 |
|---|---|---|---|
| `aggressive` 激进 | 通用假设优先 | 30% | 2 条（草率但抢先） |
| `conservative` 保守 | 汤面细节优先 | 85% | 全部（扎实但慢） |
| `balanced` 均衡 | 交替 | 60% | 5 条 |
| `creative` 跳跃 | 完全打乱 | 55% | 全部（方差大） |

## 对抗模式 · 欺骗（deception）

`mode="deception"`：**首位玩家为知情者**，掌握真相并可撒谎；其余为求解者。
回合只在求解者之间轮转，知情者只负责应答。

| 知情者策略 | 行为 |
|---|---|
| `insider-honest` | 始终如实 —— **对照组**，衡量"没人捣乱时求解者能考多少" |
| `insider-misleading` | 始终反向 —— 激进欺骗，但规律明显 |
| `insider-mixed` | 50% 概率撒谎 —— 真话假话混着说，最难对付 |

实测（12 题 × 保守 + 激进两种求解者）：

| 知情者 | 撒谎率 | 求解者平均命中率 |
|---|---|---|
| honest（对照） | 0% | **0.243** ← 基准线 |
| mixed | 47% | 0.104 |
| misleading | 100% | **0.000** |

结论：**撒谎率与求解者命中率强负相关**，欺骗能力首次成为可量化指标。
反直觉的一点：全程撒谎反而"规律过于明显"（求解者反着听即可），
实战中真话假话混着说才最难对付。

> 安全设计：撒谎记录 `_lie_log` 会落盘供离线评测，但**不进入玩家可见状态**，
> 否则求解者查一下日志就能看穿所有谎言。

### 质疑与伪装

求解者可以**重复提问**来测谎：同一个问题问两次，回答不一致就发起 `challenge` 指认谎言。
复核倾向 `repeat_rate`：保守 0.30 / 均衡 0.15 / 跳跃 0.10 / 激进 0.05。

| 知情者 | 撒谎率 | 被抓率 | 未识破谎言 | 伪装分 | 求解者命中率 |
|---|---|---|---|---|---|
| honest | 0% | 0% | 0.0 | N/A | 0.306 |
| mixed | 50% | 29% | 4.8 | 0.71 | 0.222 |
| misleading | 100% | 0% | 14.8 | 1.00 | **0.000** |

**伪装**由此第一次有了精确定义：**撒了谎却没被抓到**。

两条值得注意的结论：

- **复核倾向真实转化为识破能力**：保守派平均质疑 1.9 次，激进派只有 0.1 次（差 19 倍）。
- **misleading 一次都没被抓** —— 它稳定说反话，重复问答案完全一致。
  这是求解者当前的**真实能力局限**（破解它需要"反着听"的策略），不是 bug。
  也因此暴露一点：伪装分高 ≠ 欺骗成功，两个指标必须分开看。

## 对抗模式 · 辩论（debate）

`mode="debate"`：两名求解者就同一汤面**互相比较论证**，收网动作从 `guess`（猜想）
变为 `argue`（论证）。

与问答模式的根本区别是**比较制**：问答要求达到绝对命中阈值（0.6）才算赢；
辩论不看绝对阈值——只要一方的论证比对手更站得住脚（命中更多关键事实）即胜出，
这正是"辩论"对抗性的体现。

| 指标 | 含义 |
|---|---|
| `argument_accuracy` | 各自论证命中了多少关键事实（论证质量） |
| `persuasion` | 己方命中率 / 全场最佳，1.0 = 本轮最有说服力 |
| `evidence_used` | 论证前提问命中多少事实（论据充分度） |
| `ranking` | 按论证质量排名 |

**平分加时赛**：命中率并列时，比论据充分度（各自 yes 提问数）——论据越多越可验证；
副标准仍完全持平则判**无人胜出**（`winner=None`），绝不按出手顺序送胜。

实测（4 性格两两对战 × 4 题，24 场）：激进 9 胜 / 跳跃 9 胜 / 均衡 5 胜 / 保守 1 胜，
0 平局。覆盖率制评分下"话多反而摊薄分数"——堆砌全部线索不如精简命中，
保守派广撒网的风格在辩论里最吃亏。

## 评测维度

**四维推理画像**（问答模式，供雷达图）：

| 维度 | 含义 | 实测最突出者 |
|---|---|---|
| `accuracy` 推理准确 | 猜想命中率 —— 说得对不对 | 均衡 / 跳跃 0.250 |
| `efficiency` 信息效率 | 每次提问命中多少关键事实 —— 问得准不准 | **激进 0.219** |
| `breadth` 命中广度 | 触及的关键事实占全题比例 —— 挖得广不广 | **保守 0.361** |
| `decisiveness` 收网果断 | 多早就敢提交猜想 —— 敢不敢下结论 | **激进 0.625** |

一句话读图：激进派"问得最准、收网最快，但结论草率"；保守派"挖得最广，却最不敢下结论"。

**欺骗指标**（欺骗模式）：`deception_rate` 撒谎率、`avg_solver_accuracy` 求解者在干扰下的命中率。

**辩论指标**（辩论模式）：`argument_accuracy` 论证质量、`persuasion` 说服力、`evidence_used` 论据充分度。

> ⚠️ **诚实边界**：项目目标是推理 / 欺骗 / 伪装 / 辩论四维，v0.5 起
> **推理 ✅、欺骗 ✅、伪装 ✅、辩论 ✅**（比较制下的说服力，可精确测量）。
> 注意辩论维度当前由启发式玩家驱动，接入真实 LLM 后论证文本才会真正分化。

## 公平性设计

- **轮末统一结算**：玩家猜中不会立刻终结对局，而是记入本轮待结算队列，
  等所有玩家都行动完后统一比较，命中率最高且达标者胜。避免"先手靠手速碾压"。
- **平分加时赛**：分数并列时不按出手顺序送胜——辩论比论据充分度、问答比提问效率，
  副标准也持平才判无人胜出。避免"先手按字典序白赢"。
- **角色隔离**：`mock`（作弊）与 `insider`（非求解者）默认不进入求解者排名。

## 题库

内置 **12 道**海龟汤，覆盖 `easy` / `medium` / `hard` 三档，标签含悬疑、推理、反转、
温情、密室、法律、科幻等。

## 接入 MCP 客户端（Claude Desktop / CodeBuddy 等）

```json
{
  "mcpServers": {
    "ai-arena": {
      "command": "python",
      "args": ["/workspace/mcp_server/server.py"]
    }
  }
}
```

## Tools（工具）

| 工具 | 作用 |
|---|---|
| `list_puzzles` | 列题库（不含答案） |
| `get_puzzle` | 取题（`include_truth` 默认 False，玩家禁止取真相） |
| `start_match` | 开一局，支持 `mode="deception"` / `mode="debate"` |
| `get_match_state` | 看当前状态/对话/比分 |
| `step_match` | 推进一步（可中途介入） |
| `run_match` | 一键跑通整局 |
| `list_matches` | 列历史对局 |
| `analyze_match` | 单局能力分析（欺骗模式给欺骗指标，辩论模式给辩论指标） |
| `get_leaderboard` | 跨模型排名（默认剔除作弊者与知情者） |
| `get_profiles` | 四维推理画像 |
| `export_report` | 生成排行榜图 + 雷达图 + 文字报告 |

## Resources / Prompts

- 资源：`arena://puzzles`、`arena://matches/{match_id}`、`arena://models`
- 提示：`player_turn`、`judge_answer`、`debrief`（与引擎内部 prompt 同源）

## 配置 API Key

复制 `.env.example` 为 `.env` 并填入对应厂商 key，即可把 `heuristic-*` 换成真实模型名
（如 `gpt-4o-mini`、`deepseek-chat`、`qwen-plus`、`kimi`）。

## 测试

```bash
pip install pytest && pytest      # 45 项，覆盖不作弊、有区分度、结算公平性、欺骗/质疑/辩论
```

## 路线图

- v0.1：题库 + 引擎 + Mock 玩家 + MCP 骨架 + 单局跑通 ✅
- v0.2（部分）：启发式玩家 + 题库 12 题 ✅；真实多模型对战、LLM 裁判 ⏳
- v0.3：轮末统一结算、四维画像、雷达图、角色隔离 ✅
- v0.4：deception 欺骗模式、知情者三策略、challenge 质疑动作、欺骗/伪装指标 ✅
- v0.5（当前）：debate 辩论模式（比较制 + 平局加时赛 + 辩论指标）、风格名冒号解析修复 ✅
- v0.6：破解 misleading 的"反向利用"求解者策略、HTTP/SSE 传输
