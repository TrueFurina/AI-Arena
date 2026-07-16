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

## 待办 / 路线图
- v0.2：真实多模型对战（litellm，填 .env key）、对抗式辩论/欺骗模式、LLM 裁判
- v0.3：能力指标深化（推理/欺骗/伪装/辩论四维）、排行榜增强、多维度雷达图
- v0.4（可选）：HTTP/SSE 传输、资源订阅、小红书风格报告导出

## 环境备注
- 沙箱 Python 3.11，`pip install -e .` 装 `mcp` 1.28.1 + pandas/matplotlib；`fastmcp` 3.4.4 作为测试客户端（dev 依赖）。
- `tests/test_server.py` 用 `fastmcp.Client(srv.mcp)` 内存模式验证 MCP 层，规避 stdio 手动解析的版本差异。
