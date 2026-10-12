# AI-Arena 扩展指南（EXTENDING）

> 当你要给系统「加东西」时，按本文操作。覆盖三类最常见扩展：**加模型**、**加题库题目**、**加游戏模式**。

---

## 0. 总原则

- **零 key 优先**：新增的真实模型必须有 mock/规则化退路，保证无 key 也能跑。
- **配置不写死**（规则 2）：模型名/key 走 `.env` 与别名映射，不进业务代码。
- **日志字段向后兼容**：`build_log()` 的 `result` 新增字段时，旧日志缺字段要能补 0。
- **先写 skill/doc**：重复性扩展操作顺手沉淀成 skill（见 `.codebuddy/skills/`）。

---

## 1. 新增一个 LLM 供应商 / 模型

### 1.1 绝大多数情况：litellm 已原生支持
DeepSeek / Qwen / Kimi / OpenAI / Claude 等主流模型，**无需改代码**，只需：
1. 在 `.env` 填对应 key：

   | 供应商 | env 变量 | `--model` 别名 | litellm 模型名 |
   |---|---|---|---|
   | DeepSeek | `DEEPSEEK_API_KEY` | `deepseek` | `deepseek/deepseek-chat` |
   | 通义千问 | `DASHSCOPE_API_KEY` | `qwen` | `qwen/qwen-plus` |
   | Kimi | `MOONSHOT_API_KEY` | `kimi` | `moonshot/moonshot-v1-8k` |
   | OpenAI | `OPENAI_API_KEY` | `gpt` | `openai/gpt-4o-mini` |
   | Claude | `ANTHROPIC_API_KEY` | `claude` | `anthropic/claude-3-5-sonnet-20241022` |

2. 跑：`python run/single_game.py --mode llm --model <别名> --gm smart --puzzle turtle_001`
3. 先用 `python run/test_api.py --model <别名>` 验证连通性。

### 1.2 加一个「友好别名」
若想给常用模型起短名（如 `my-model`），编辑 `players/llm_player.py` 的 `MODEL_ALIASES`：
```python
MODEL_ALIASES = {
    ...,
    "my-model": "provider/model-name",   # 完整 litellm 名
}
```
并在 `run/test_api.py` 的 `ALIAS_ENV` 里补上它需要的 env 变量名，自检工具才会识别。

### 1.3 接入「自定义 OpenAI 兼容端点」
自有/第三方网关（返回 OpenAI 格式）：
```
# .env
OPENAI_API_BASE=https://your-gateway.example.com/v1
OPENAI_API_KEY=sk-xxxx
```
然后 `--model openai/<模型名>` 即可，无需改码。

### 1.4 新增「非 litellm 原生」供应商
若供应商 litellm 不直接支持：
1. 在 `players/llm_player.py` 写一个薄适配（复用 `act()` / `_parse_action()`），把返回规范成 `{"action","content"}`；
2. key 走 `.env` 新变量；
3. 在 `run/test_api.py` 把该供应商加进自检列表；
4. 严禁在代码里硬编码 key（规则 2）。

> ⚠️ 接入前先用最小 curl 验证 `.../chat/completions` 可达。tokenc.top 教训：很多「聚合网关」只提供翻译/天气等工具 API，**不提供 LLM 对话**，别白忙。

---

## 2. 新增题库题目

题库在 `engine/puzzles.py`，`get_puzzle(id)` 按 id 取题。每题字段：

```python
{
  "id": "turtle_009",
  "surface": "汤面：一句话情境（提问者能看到）",
  "bottom": "汤底：完整真相（GM 私有，不泄漏给提问者）",
  "key_facts": ["事实1", "事实2", "..."],        # 还原度判定的「真相骨架」
  "clue_gradient": ["梯度线索1", "..."],          # 部分相关信息（yes_and_no）
  "keywords": {                                    # M2：关键事实 -> 触发词（容错改写）
      "事实1": ["词A", "词B"],
      "事实2": ["词C"],
  },
  "clue_keywords": {                               # M2：梯度线索 -> 触发词
      "梯度线索1": ["词D"],
  },
}
```

### 2.1 字段规则
- `key_facts` / `clue_gradient`：保持**字符串列表**，向后兼容旧日志与指标。
- `keywords` / `clue_keywords`：**dict，键必须出现在上面的列表里**，值是该事实的「触发关键词」列表。
- SmartGM（`players/gm.py`）判定逻辑：
  - 提问含某 `key_facts` 的任一 `keywords` → 答 `yes`；
  - 提问含某 `clue_gradient` 的任一 `clue_keywords` → 答 `yes_and_no`；
  - 否则 `irrelevant`。
- 回退：若某事实没配 `keywords`，SmartGM 用事实原文本身当触发词（仍可用，但容错差）。

### 2.2 出题 checklist
- [ ] `surface` 只给情境、不泄底；
- [ ] `bottom` 能由 `key_facts` 拼回（事实是汤底的骨架）；
- [ ] 每个 `key_facts` 至少配 1–3 个 `keywords`，覆盖模型的**常见自由表述**（如事实「男人打嗝」→ 关键词 `["打嗝","打咯"]`）；
- [ ] `clue_gradient` 放「相关但不核心」的线索；
- [ ] 用 `arena-puzzle` skill 或手动跑一次 `--gm smart` 验证还原度能被合理探测。

---

## 3. 新增游戏模式（玩法 A/B/C/D）

当前只有 **A. 单 GM + 单提问者**。扩展新玩法时：

### 3.1 不动 `engine/game.py` 的核心
`Game` 是「单 GM + 单提问者」回合机。新玩法优先用**组合**而非重写：
- **B. 双盲互猜**：两个 `LLMQuestioner` 互为 GM/提问者，各自持有一半真相，复用 `Game` 跑两遍；
- **C. 骗子变体**：给 `LLMQuestioner` 加 `role=liar/truthful` 参数，GM 按角色决定回答真伪；
- **D. 辩论裁决**：在 `Game` 结束后加一个 `debate` 阶段，引入 `LLM-as-Judge`（`players/judge.py`）。

### 3.2 新模式接入点
- 玩家抽象在 `players/base.py`（`Player` / `GM`），新角色继承它；
- 动作协议统一 `{"action","content"}`，新角色也走这套，保证日志结构一致；
- 若新模式产生新指标，在 `eval/metrics.py` 加函数，并在 `build_log()` 的 `result` 加字段（向后兼容）。

### 3.3 四维度落点
| 玩法 | 维度 | 关键改造 |
|---|---|---|
| A | 推理 | 已有 |
| B | 推理+伪装 | 双 `Game` 互探 |
| C | 欺骗/伪装 | 角色参数 + GM 真伪分支 |
| D | 辩论 | 新增 debate 阶段 + Judge |

---

## 4. 扩展后别忘了

1. **测试**：零 key 下用 mock/假模型注入验证新逻辑（`tests/`）；
2. **skill/doc**：新操作若重复出现，加 skill（`arena-*`）或补 `doc/`；
3. **MEMORY**：大版本扩展更新 `.workbuddy/memory/MEMORY.md`（规则 3）；
4. **向后兼容**：旧 `logs/` 仍能 `evaluate.py` 聚合（缺字段补 0）。
