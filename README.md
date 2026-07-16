# AI 角斗场 (Arena) · MCP 服务器 (v0.1)

让不同的 AI 模型在海龟汤（社交推理游戏）里相互对抗，通过对局数据评估各模型的
**推理 / 欺骗 / 伪装 / 辩论** 能力。本仓库是评测系统的 **MCP 服务器**，把游戏引擎 +
多模型对战 + 自动评测封装成标准 MCP 工具，供任意支持 MCP 的客户端调用。

> 本质：AI vs AI 的离线评测系统，不是真人游戏产品。
> v0.1 以 **Mock 玩家** 离线跑通整条流程，无需任何 API key。

## 快速开始

```bash
pip install -e .            # 安装本体（依赖 mcp / pydantic / pandas / matplotlib）
python run.py               # 以 stdio 方式启动 MCP 服务器
```

不接 key 也能跑：

```python
from mcp_server import server as srv
mid = srv.start_match("ts-001", ["mock", "mock"], max_rounds=20)["match_id"]
srv.run_match(mid)          # 非交互跑完整局
srv.get_leaderboard()       # 看排名
```

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
| `start_match` | 开一局，返回 `match_id` |
| `get_match_state` | 看当前状态/对话/比分 |
| `step_match` | 推进一步（可中途介入） |
| `run_match` | 一键跑通整局 |
| `list_matches` | 列历史对局 |
| `analyze_match` | 单局能力分析 |
| `get_leaderboard` | 跨模型排名 |
| `export_report` | 生成图表+文字报告 |

## Resources / Prompts

- 资源：`arena://puzzles`、`arena://matches/{match_id}`、`arena://models`
- 提示：`player_turn`、`judge_answer`、`debrief`（与引擎内部 prompt 同源）

## 配置 API Key

复制 `.env.example` 为 `.env` 并填入对应厂商 key，即可把 `"mock"` 换成真实模型名
（如 `gpt-4o-mini`、`deepseek-chat`、`qwen-plus`、`kimi`）。真实多模型对战在 v0.2 深化。

## 测试

```bash
pip install pytest && pytest
```

## 路线图

- v0.1（当前）：题库 + 引擎 + Mock 玩家 + MCP 骨架 + 单局跑通
- v0.2：真实多模型对战（litellm）、对抗式辩论/欺骗模式
- v0.3：能力指标深化、排行榜、pandas/matplotlib 报告
- v0.4（可选）：HTTP/SSE 传输、资源订阅、小红书风格报告
