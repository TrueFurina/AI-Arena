# AI Arena · 模型横向对决报告（deepseek-chat vs glm-4-flash）

- 生成时间：2026-10-12
- 协议：2 模型 × 3 题（turtle_soup_001/002/003，均含 golden_qa）= 6 局
- 成本：**deepseek × 3 局 ≈ ¥0.045**（按 10 次调用/局、≈¥0.015/局估算）；**glm-4-flash 免费**（官方额度）
- 总耗时：101.6s
- 复现：`COMPARE_MODELS="deepseek/deepseek-chat,zai/glm-4-flash" COMPARE_PUZZLES="turtle_soup_001,turtle_soup_002,turtle_soup_003" python web-proto/run_compare.py`

## 指标定义（全部自动从对局 transcript/scores 提取）

| 指标 | 含义 | 备注 |
|---|---|---|
| 有效提问率 vra | 提问以问号结尾或含疑问词的比例 | 弱模型退化为句号/陈述句/空响应会拉低 |
| 主持 yes/no 率 | 主持回答含「是/否/对/不对」等的比例 | 答非所问会拉低 |
| 还原 F1 | 终局还原 vs 汤底的字符级集合 F1 | **客观指标**，不依赖裁判模型 |
| 裁判评分 | LLM 裁判给的 0–100 分 | ⚠️ 同模型自评，存在自偏好偏差 |

## 汇总（各模型 3 题均值）

| 模型 | 有效提问率 | 主持yes/no率 | 还原F1 | 裁判评分 |
|---|---|---|---|---|
| deepseek/deepseek-chat | 100% | 100% | **0.53** | 78.3 |
| zai/glm-4-flash | 100% | 58% | **0.29** | 68.3 |

## 逐题明细

| 模型 | 题目 | 有效提问 | yes/no率 | 还原F1 | 评分 |
|---|---|---|---|---|---|
| deepseek/deepseek-chat | turtle_soup_001 | 4/4 | 100% | 0.61 | 55.0 |
| deepseek/deepseek-chat | turtle_soup_002 | 4/4 | 100% | 0.60 | 85.0 |
| deepseek/deepseek-chat | turtle_soup_003 | 4/4 | 100% | 0.38 | 95.0 |
| zai/glm-4-flash | turtle_soup_001 | 4/4 | 75% | 0.35 | 60.0 |
| zai/glm-4-flash | turtle_soup_002 | 4/4 | 75% | 0.25 | 60.0 |
| zai/glm-4-flash | turtle_soup_003 | 4/4 | 25% | 0.27 | 85.0 |

## 结论

1. **客观差距明确**：还原 F1 上 deepseek 0.53 vs glm-4-flash 0.29（近一倍），是硬证据——付费模型在「从 yes/no 线索重建真相」上明显更强。裁判评分（78.3 vs 68.3）方向一致，但因是同模型自评、存在自偏好偏差，**以 F1 为准**。

2. **提问能力已拉平**：两模型有效提问率均为 100%（各 3 题 12/12 全有效）。这与早期诊断（glm-4-flash 在 `_transcript_to_msgs` 路径下 R3 起退化至 25%）形成对照——说明 `players/llm.py` 的两轮 prompt 加固（去方括号标签 + 问号约束）在**正式提问路径**上显著缓解了免费模型退化。

3. **免费模型的真实短板在「主持」环节**：glm-4-flash 主持 yes/no 率仅 58%（第 3 题 25%），即它在「给 yes/no 答复」时答非所问（输出解释而非判断）；deepseek 为 100%。这比提问退化更难靠 prompt 弥补。

4. **证据等级**：每模型 3 题小样本，方向明确但**未达统计显著**；结论为初步量化，非最终定论。真实账单以 deepseek 后台为准。

## 产物

- 可复现脚本：`web-proto/run_compare.py`
- 原始对局：各 `web-proto/logs/compare_<model>_<pid>.json`
- 本地图表：`web-proto/logs/compare_chart.png`（4 指标并排柱状图）
- 本地报告：`web-proto/logs/compare_report.md`
