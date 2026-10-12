# 密钥与 API 连接指南（KEYS）

> 如何安全地接入模型 API。覆盖：`.env` 管理、各供应商对照、`run/test_api.py` 自检、自定义端点，以及**本次踩过的 tokenc.top 坑**。

---

## 0. 铁律（规则 2）

> **任何 API key / 端点都只放 `.env`，代码里绝不写死。**

- `.env` 不入库（gitignore）；
- `.env.example` 入库当模板（不含真实值）；
- litellm 按环境变量自动读取各供应商 key，业务代码不出现 key 字符串。

---

## 1. `.env` 结构

复制模板并填值：
```bash
cp .env.example .env
# 编辑 .env，填入真实 key
```

当前 `.env` 关键变量：
```
# 模型供应商
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=     # 通义千问 Qwen
MOONSHOT_API_KEY=      # Kimi

# 自定义 OpenAI 兼容端点（可选）
# OPENAI_API_BASE=https://your-gateway.example.com/v1
# OPENAI_API_KEY=sk-xxxx

# tokenc.top 网关（仅 run/test_api.py 联调探测用，非 LLM 后端）
TOKENC_API_KEY=
```

---

## 2. 供应商 ↔ 别名 ↔ 环境变量 对照

| 供应商 | env 变量 | `--model` 别名 | litellm 模型名 |
|---|---|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek` | `deepseek/deepseek-chat` |
| 通义千问 Qwen | `DASHSCOPE_API_KEY` | `qwen` | `qwen/qwen-plus` |
| Kimi（月之暗面） | `MOONSHOT_API_KEY` | `kimi` | `moonshot/moonshot-v1-8k` |
| OpenAI | `OPENAI_API_KEY` | `gpt` | `openai/gpt-4o-mini` |
| Claude | `ANTHROPIC_API_KEY` | `claude` | `anthropic/claude-3-5-sonnet-20241022` |

调用示例：
```bash
python run/single_game.py --mode llm --model deepseek --gm smart --puzzle turtle_001
```

---

## 3. 连通性自检（必做）

接入任何 key 后，先跑自检，避免「跑到第 10 轮才发现 key 错了」：
```bash
python run/test_api.py                 # 测全部已配置供应商 + tokenc（若设 key）
python run/test_api.py --model deepseek # 只测某个
python run/test_api.py --skip-tokenc    # 不测 tokenc
python run/test_api.py --timeout 15     # 自定义超时
```

行为：
- 配了 key 的供应商：发一个最小 chat 请求，报告 `OK`（含延迟/首句）/`FAIL`（含原因：鉴权失败/限流/超时/参数错误）；
- 没配 key：优雅 `SKIP`（退出码 0，不报错）；
- tokenc：探测 `translate`（存活性）与 `chat`（预期不可用），自动给结论。

自检实现见 `run/test_api.py`（读 `.env` → litellm 发请求 → 分类报告）。

---

## 4. 自定义 OpenAI 兼容端点

自有网关或第三方聚合（返回 OpenAI 格式）：
```
# .env
OPENAI_API_BASE=https://your-gateway.example.com/v1
OPENAI_API_KEY=sk-xxxx
```
```bash
python run/single_game.py --mode llm --model openai/<模型名> --gm smart
```
无需改业务代码。

---

## 5. ⚠️ tokenc.top 实测结论（重要坑）

用户曾提供 `ag_...` 令牌，期望 tokenc.top 当模型后端。实测结论：

- **令牌有效**，但 tokenc.top 是「**通用工具 API 聚合网关**」（翻译/天气/身份证等），**不提供 LLM 对话能力**；
- 探测结果：

  | 路径 | 结果 |
  |---|---|
  | `/v1/translate` | ✅ `success:true`（网关本身通） |
  | `/v1/chat/completions` | ❌ 404 `PROVIDER_NOT_FOUND`（"服务 chat 暂未接入"） |
  | `/v1/llm` `/v1/openai` `/v1/ai` `/v1/models` | ❌ 同上，全部「暂未接入」 |

- **结论**：tokenc.top **不能**作为 AI-Arena 的模型后端。它的 `TOKENC_API_KEY` 仅用于 `run/test_api.py` 联调探测演示。

> 通用教训：**不是所有「API 聚合平台」都提供 LLM**。接入新供应商前，用最小 curl 验证 `.../chat/completions` 能返回正常 completion，再写进配置。

---

## 6. 运行环境提醒

- 系统 Python 受 PEP 668 保护时，用隔离 venv（如 `/root/venv_aiarena`）装 `litellm` / `pandas` / `matplotlib` / `python-dotenv` / `pydantic`，**不要** `--break-system-packages`；
- 跑命令前确保用该 venv 的 python：`/root/venv_aiarena/bin/python run/test_api.py`；
- 国内访问海外端点不稳定时，可设 `LITELLM_PROXY_URL` 走代理。

---

## 7. 安全与健康

- key 贴进对话/`.env` 后，验证完毕可在供应商后台**轮换/吊销**，不影响本次结果；
- 不要把 `.env` 提交进任何仓库或同步到项目盘；
- 多模型对比时，每个模型的 key 各自独立，便于单独吊销某一家而不影响其他。
