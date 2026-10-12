# M5 骗子变体（Deception Variant）设计规格

> 维度定位：**A 推理维度**的对抗延伸 —— 在「提问者探测真相」之上，引入「信息可能不可信」的博弈层。
> 对应 `doc/VISION.md` 四维度中的 **C 骗子变体**。

---

## 1. 动机与目标

M0–M3 的默认设定是「GM 永远诚实」。但真实推理有两个被忽略的维度：

1. **信息可靠性**：人类在玩海龟汤 / 狼人杀 / 辩论时，面对的回答未必全真。模型能否在「有人撒谎」时仍收敛到真相，是比「诚实环境还原度」更强的推理能力信号。
2. **主动检测**：强推理者不会盲目相信每个 yes/no，而会**交叉验证、寻找矛盾**。当前指标只衡量「探测到多少事实」，无法区分「碰巧问对」与「识破谎言后仍问对」。

M5 目标：在不破坏 M0–M3 引擎的前提下，新增一层**可控欺骗**，并产出可量化的**鲁棒性指标**。

---

## 2. 三种变体（从易到难）

| 变体 | 撒谎方 | 难度 | 状态 |
|------|--------|------|------|
| **V1 Deceiver GM** | GM 偶尔翻转 yes/no | 低 | ✅ PoC 已落地（`players/gm.py::DeceiverGM`） |
| **V2 Adversarial Questioner** | 提问者故意误导 GM（红队） | 中 | 设计就绪，未实现 |
| **V3 Impostor** | 多提问者中混入卧底，互证真伪 | 高 | 设计就绪，未实现 |

**本期只做 V1**（已实现可运行 PoC），V2/V3 留给后续，避免在真实 key 就绪前过度投资（先度量后优化）。

---

## 3. 协议扩展

### 3.1 GM 回答加 `lied` 字段（已落地）

```json
// 诚实回答
{"answer": "yes"}
// 撒谎回答（仅翻转 yes<->no；irrelevant/yes_and_no/unknown 不撒谎）
{"answer": "no", "lied": true}
```

- `lied` **只写进历史与日志，绝不进观察**（提问者看不到自己被骗，否则指标失真）。
- 向后兼容：旧日志无 `lied` 字段，`engine/game.py` 统计时默认 0。

### 3.2 提问者 `challenge` 动作（规划中，V2 用）

```json
{"action": "challenge", "content": "你刚才说 X 是 no，但第 3 轮又说 Y 是 yes，二者矛盾"}
```

GM 收到 challenge 需返回 `{ "answer": "uphold" | "retract", "lied": <bool> }`，用于度量 **lie_detection_rate**。本期不实现，先验证引擎层（V1）通畅。

---

## 4. 指标定义

| 指标 | 定义 | 已实现 |
|------|------|--------|
| `lie_rate` | 实际撒谎轮数 / 已回答轮数（`len(history)` 为分母，不含最后 guess 轮） | ✅ |
| `deception_robustness` | 撒谎环境还原度 / 诚实环境还原度 ∈ [0,1]；=1 表示撒谎无影响，=0 表示一撒谎就崩 | ✅（`eval/metrics.py`） |
| `lie_detection_rate` | 被提问者通过 challenge 成功识破的谎言占比 | ⬜ V2 |

> 鲁棒性用「同提问者跨环境对比」而非单局绝对值，避免把「模型本身弱」误读成「被欺骗」。

---

## 5. 架构与代码落点

```
players/gm.py
  └─ class DeceiverGM(SmartGM)        # 继承 SmartGM，按 lie_rate 翻转 yes/no
engine/game.py
  └─ Game.run()                       # 记录 resp.get("lied") 进 history
  └─ Game.build_log()                 # result.deception = {lies_total, lie_rate}
eval/metrics.py
  ├─ metrics_from_log()               # 新增 lies_total / lie_rate 列
  ├─ aggregate_by_model()             # 新增 mean_lie_rate
  ├─ mean_lie_rate(rows)
  └─ deception_robustness(r_truth, r_lie)
run/deceiver_demo.py                  # 零 key 演示：同提问者跨诚实/撒谎环境对比
```

**零 key 验证路径**（已跑通）：

```bash
python run/deceiver_demo.py                 # lie_rate=0.3, seed=42
# 输出每题 诚实还原度 / 撒谎还原度 / 鲁棒性 / 实际撒谎率
```

实测（seed=42）：诚实还原度 1.00 → 撒谎环境 0.25–0.50，鲁棒性 0.25–0.50。
因当前提问者是「照念关键事实」的 mock（无矛盾检测），鲁棒性偏低属预期；
真实模型若具备检测能力，鲁棒性应显著高于此基线 —— 这正是 M5 要衡量的差异。

---

## 6. 与 tournament 的集成（规划）

`config.toml` 未来可加：

```toml
[globals]
mode = "deceiver"          # single | deceiver
lie_rate = 0.3

[[models]]
alias = "deepseek"
litellm = "deepseek/deepseek-chat"
enabled = false
```

`run/tournament.py` 在 `mode="deceiver"` 时把 `DeceiverGM(lie_rate)` 注入 `_build_gm()`，
并在聚合时对比同模型的 `restoration`（诚实）与 `restoration_under_lies`（撒谎），自动算鲁棒性。

---

## 7. 反模式（避免）

- ❌ 把 `lied` 泄进观察 —— 指标会失真，测不出真实鲁棒性。
- ❌ 用 `self.turn` 当 `lie_rate` 分母 —— 含最后 guess 轮（无 GM 回答），虚高撒谎率（已修正为 `len(history)`）。
- ❌ 在非 yes/no 答案上撒谎 —— 对 irrelevant 撒谎无信息量，退化为纯噪声。
- ❌ 单局绝对值当鲁棒性 —— 必须「同提问者跨诚实/撒谎」对比，隔离模型本身强弱。
