# M6 辩论 + LLM-as-Judge 设计规格

> 维度定位：**D 辩论维度** —— 从「单人推理」升级到「多 Agent 立场对抗 + 自动裁判」，衡量说服力、逻辑一致性与对抗鲁棒性。
> 对应 `doc/VISION.md` 四维度中的 **D 辩论 + Judge**。

---

## 1. 动机与目标

M0–M5 都是「单提问者 vs 单 GM」的**单人推理**。M6 把评测升维：

1. **对抗产生信号**：让模型为对立立场辩护，比「自问自答」更能暴露推理深度与盲区。
2. **可扩展裁判**：用 LLM 当裁判（LLM-as-Judge），把「谁更有道理」变成可复现、可审计的分数，而非主观印象。
3. **贴近真实能力**：现实中的 AI 协作/争论场景（如 agent 协商、红蓝对抗）本质是「辩论」。

M6 目标：在零 key 可跑的框架上，定义清晰的辩论状态机、裁判协议与指标，并预留真实 LLM 接入点。

---

## 2. 架构（已落地零 key PoC）

```
engine/debate.py
  ├─ class Debater                 # 基类：take_turn(observation) -> {text}
  ├─ class MockDebater(Debater)    # 零 key：按立场循环取预置论点
  ├─ class Judge                   # 基类：score(transcript) -> verdict
  ├─ class MockJudge(Judge)        # 零 key：证据密度 + 篇幅 启发式打分
  └─ class Debate                  # 状态机：命题 → 多轮正反发言 → 裁判
run/debate_demo.py                 # 零 key 演示 runner
```

**依赖注入（与 LLMQuestioner 同范式）**：真实 LLM 接入点通过 `completion_fn` 预留。
零 key 用 `MockDebater` / `MockJudge` 跑通 pipeline；拿到 key 后换 `LLMDebater` / `LLMJudge`，接口不变。

---

## 3. 协议

### 3.1 给辩论者的观察
```json
{
  "proposition": "AI 应被允许自主参与海龟汤评测",
  "stance": "pro" | "con",
  "round": 1,
  "history": [{"speaker": "...", "stance": "...", "round": 1, "text": "..."}]
}
```

### 3.2 裁判 verdict（结构化、可聚合）
```json
{
  "winner": "pro",
  "scores": {"pro": 7.855, "con": 3.772},
  "rationale": "裁判（启发式）判 pro 胜，分差 4.083。pro: 证据词 7 / 篇幅 114 字；con: ..."
}
```

---

## 4. 指标定义（规划）

| 指标 | 定义 | 状态 |
|------|------|------|
| `persuasion_score` | 单方的裁判得分（证据密度 + 立场一致性 + 篇幅） | ✅ MockJudge |
| `persuasion_margin` | `|pro - con|`，分差越大立场优势越明显 | ✅ demo 输出 |
| `consensus` | 多裁判 / 多次复跑的一致率（稳健性） | ⬜ |
| `rebuttal_effect` | 对方发言后本方得分变化（是否被驳倒） | ⬜ |
| `judge_agreement` | LLM 裁判与人类裁判标注的吻合度（防裁判偏见） | ⬜ |

---

## 5. 真实 LLM 接入（设计就绪，未实现）

### 5.1 LLMDebater
```python
class LLMDebater(Debater):
    def __init__(self, model, stance, completion_fn=None, **kw):
        self.model = model; self.stance = stance
        self.completion_fn = completion_fn  # 注入 litellm.completion
    def take_turn(self, obs):
        # 拼 system（立场 + 命题）+ user（历史），要求输出 JSON {text}
        ...
```

### 5.2 LLMJudge（关键：rubric 要透明）
裁判 prompt 必须明示评分维度，避免黑箱：
```
你是辩论裁判。对 below 的 transcript 按三维度打分（各 0-10）：
1. 逻辑一致性（是否自洽、有无矛盾）
2. 证据强度（是否引用事实/数据/机制，而非空泛断言）
3. 切题度（是否回应命题与对方论点）
输出 JSON：{"winner":"pro|con","scores":{"pro":float,"con":float},"rationale":str}
```
> ⚠️ LLM 裁判有位置偏见（常偏好先发言方）/ 冗长偏见。缓解：随机化发言顺序、多次复跑取 consensus、与人工标注算 judge_agreement。

---

## 6. 零 key 验证路径（已跑通）

```bash
python run/debate_demo.py                 # 默认命题 + 3 轮
python run/debate_demo.py --rounds 4
# 输出逐轮发言 + 裁判结论（winner / scores / rationale）+ 说服力分差
```

实测（3 轮）：正方胜，pro=7.855 / con=3.772，分差 4.083。
因论点池是手写的，分差仅演示 pipeline；**真实信号来自 LLMDebater 的自由对抗**。

---

## 7. 与 tournament 的集成（规划）

`config.toml` 未来可加：

```toml
[globals]
mode = "debate"
rounds = 3

[[debates]]
proposition = "AI 应被允许自主参与海龟汤评测"
pro = "deepseek"
con = "qwen"
judge = "gpt"
```

`run/tournament.py` 在 `mode="debate"` 时调用 `Debate(...)` 而非 `Game(...)`，聚合 `persuasion_margin` / `consensus`。

---

## 8. 反模式（避免）

- ❌ 黑箱裁判 prompt —— 必须明示 rubric，否则分数不可审计、不可复现。
- ❌ 单次裁判定胜负 —— LLM 有偏见，至少多次复跑取 consensus。
- ❌ 忽略发言顺序偏见 —— 先发言方常被高估，要随机化顺序。
- ❌ 在零 key 下过度投资 LLMDebater —— 先验证 MockJudge pipeline（已做），再接真实模型。
