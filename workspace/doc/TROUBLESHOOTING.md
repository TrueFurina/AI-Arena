# 故障排查（TROUBLESHOOTING）

> 跑 AI-Arena 时常见坑与对策。零 key 优先原则：先确认 mock 链路通，再怀疑 key/网络。

---

## 1. 无 key 跑 llm 模式却「假装成功」/ 刷屏
- **现象**：没有 key 却跑了 20 轮、还原度异常。
- **原因**：真实调用被当成可重试错误反复重试。
- **正确行为**：`LLMQuestioner` 对 `AuthenticationError`/`BadRequestError` **立即上抛**，`single_game.py` 友好提示并退出码 1。
- **排查**：确认没把 key 写死在代码（规则 2）；`.env` 里真实变量名为 `DEEPSEEK_API_KEY` 等（litellm 自动读）。

## 2. tokenc.top（或某「聚合网关」）报服务未接入
- **现象**：`/v1/chat/completions` 返回 `404 PROVIDER_NOT_FOUND`（"服务 chat 暂未接入"）。
- **原因**：很多聚合平台只提供翻译/天气等**工具 API**，不提供 LLM 对话。tokenc.top 即此例（已实测）。
- **对策**：换用真正的 LLM 供应商（DeepSeek/Qwen/Kimi/OpenAI/Claude）；接入前用最小 curl 验证 `chat/completions` 可达。
- 详见 `doc/KEYS.md` §5。

## 3. matplotlib 中文显示成方框（Glyph missing）
- **现象**：图表里中文变 □。
- **原因**：默认字体缺 CJK 字形。
- **对策**：代码已注册系统 Noto Sans/Serif CJK（`eval/metrics.py::_setup_cjk_font`）；无中文字体时退化为英文标签（不掩盖真实问题）。如需中文，安装 `fonts-noto-cjk` 后重跑。

## 4. netdrive / 附件读取失败
- **现象**：调 `netdrive__tdrive` 工具报 `Expected object, received string` 或读不到 `API_KEYS.md`。
- **原因**：本会话 netdrive MCP 偶发参数解析故障；附件也不在本地 `/root/uploads`。
- **对策**：直接把 key 内容贴到对话；或重新触发附件。不要把密钥文件当代码读取路径。

## 5. pip 安装被拒（externally-managed-environment）
- **现象**：`pip install` 报 PEP 668 错误。
- **原因**：系统 Python 受保护。
- **对策**：用隔离 venv（如 `/root/venv_aiarena`），**不要** `--break-system-packages`：
  ```bash
  python3.11 -m venv /root/venv_aiarena && source /root/venv_aiarena/bin/activate && pip install -r requirements.txt
  ```

## 6. 自定义 OpenAI 兼容端点连不上
- **现象**：自建/第三方网关 401/404。
- **排查**：
  - `.env` 设 `OPENAI_API_BASE=https://your-gateway/v1` 与 `OPENAI_API_KEY`；
  - 用 `--model openai/<模型名>` 调用；
  - 先用 curl 验证 `https://your-gateway/v1/chat/completions` 可达。
- 详见 `doc/KEYS.md` §4。

## 7. 真实模型还原度普遍偏低 / 为 0
- **先别判「模型差」**。多数情况是 GM 容错不够：
  - 确认用 `--gm smart`（关键词感知），且题库配了 `keywords`/`clue_keywords`；
  - 真实模型自由改写提问时，SmartGM 靠关键词重叠判定，没配关键词就判不出；
  - 用强/弱 mock 基线（`arena-tournament --mock`）验证指标区分度（应 1.0 vs 0.0）。
- 详见 `doc/EVALUATION.md` §0。

## 8. 评测报告把旧日志也算进去了
- **现象**：`metrics.csv` 混入了上次的对局。
- **原因**：`evaluate.py` 聚合 `logs/` 下**所有** JSON。
- **对策**：限定某次范围可指定独立 `log_dir`，或先清空旧日志再跑。

## 9. 改了判定/指标逻辑但旧日志聚合崩
- **原因**：`build_log()` 新增字段未向后兼容。
- **对策**：指标模块读取旧日志时缺字段补 0（`eval/metrics.py` 已做）；新增字段要在 `result` 里给默认值。
