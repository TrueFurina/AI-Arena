# 评测体系与量化指标

> 评测是项目的「护城河」。目标：把 **推理 / 欺骗 / 伪装 / 辩论** 四个抽象能力变成可比较、可复现的数字。
> 本文在原有维度定义基础上，**补齐已实现指标的数学定义与报告阅读方式**（M2 已落地部分）。

---

## 0. 让指标「可信」的前提

没有可信指标 = 评测无意义。已落地的可信化手段：

- **SmartGM 关键词感知**（M2）：`players/gm.py` 用每题预置 `keywords` / `clue_keywords` 做重叠匹配，容错真实模型的自由改写。这样还原度反映「是否探测到关键事实」，而非「措辞是否逐字复制」。
- **过程指标与结果指标分离**：`restoration`（过程）与 `guess_score`（结果）分开算，避免「胡猜蒙对」被误判为「推理强」。
- **向后兼容**：`eval/metrics.py` 读旧日志时缺字段补 0，不崩。

> **可信度自检**：构造「强提问者」（逐事实提问）与「弱提问者」（问无关问题），在 SmartGM 下还原度应分别 ≈1.0 / ≈0.0。差距拉不开，说明指标或 GM 有问题。

---

## 1. 已实现指标（M2，规则+日志，零额外 LLM）

定义在 `engine/game.py` 与 `eval/metrics.py`：

| 指标 | 字段名 | 定义 | 含义 |
|---|---|---|---|
| **还原度** | `restoration` | 被 GM 确认(`yes`)的关键事实数 / 总关键事实数 | **推理探测过程**质量 |
| **汤底保真** | `guess_score` | 声明汤底与真实汤底的 **bigram Jaccard** | **最终陈述**与真相的相似度 |
| **效率** | `efficiency` | `restoration / turns` | 用更少轮次还原的能力 |
| **命中事实** | `key_facts_hit` | 实际命中的关键事实列表 | 可解释性（哪几条被探到） |
| **声明率** | `declared` | 是否最终 guess | 是否完成收敛 |

### 1.1 还原度 `restoration`（核心）
```python
# engine/game.py :: _judge()
for f in key_facts:
    kws = keywords.get(f, [f])          # 无关键词则回退事实原文
    if 任一历史提问含 kws 中一词 且 该轮 GM 答 yes:
        hit.append(f); break
restoration = len(hit) / len(key_facts)
```
- 关键词感知：事实「男人打嗝」配 `keywords=["打嗝","打咯"]`，模型问「他打嗝了吗」也能命中。
- 这是**推理维度**的首发量化（对应原「还原准确率」）。

### 1.2 汤底保真 `guess_score`
```python
# engine/game.py :: _guess_score()
双方相邻二元组集合 b1, b2
return len(b1 & b2) / len(b2)            # Jaccard，衡量「猜到了多少真相」
```
- 不依赖 NLP 依赖，纯字符 bigram 重合，稳定可复现。
- 与 `restoration` 分离：一个玩家可能「问得很准（restoration 高）但说不清（guess_score 低）」，或「胡猜蒙对（guess_score 高但 restoration 低）」。

### 1.3 效率 `efficiency`
`efficiency = restoration / turns`。鼓励用更少轮次还原——这是「推理经济性」的体现。

---

## 2. 四维度指标蓝图（M5/M6 进阶）

| 维度 | 首发可量化（规则+日志） | 进阶（LLM-as-Judge，M5 起） |
|---|---|---|
| **推理** | `restoration` / `guess_score` / `efficiency`（已落地） | 提问相关性 / 逻辑链质量评分 |
| **欺骗** | 谎言自洽度、误导成功率、被识破率 | 谎言说服力评分 |
| **伪装** | 人设一致性（偏离度）、露馅次数 | 角色沉浸度评分 |
| **辩论** | 说服/被说服比、证据反驳率 | 论证质量评分 |

### 各维度定义（蓝图）
- **推理**：收敛步数（越少越强）、信息增益（每问后假设空间收缩比）、还原准确率（= restoration）。
- **欺骗**（M5）：谎言自洽度（回答与自身设定一致性）、误导成功率、被识破率（越低越会骗）。
- **伪装**（M5）：人设一致性（偏离角色程度）、露馅次数（泄露真实信息次数）。
- **辩论**（M6）：说服/被说服比、证据反驳率（基于线索成功反驳对方错误主张的比例）。

---

## 3. 评测分层策略

1. **M1–M4（首发）**：只用**规则 + 结构化日志**评测，不引入第二个 LLM 当裁判，先保证跑通和可比。
2. **M5 起**：引入 **LLM-as-Judge** 做细粒度打分（需额外配置一个强裁判模型；Judge 本身应多模型交叉，避免单模型偏见）。

---

## 4. 如何运行与阅读报告

### 4.1 单局
```bash
python run/single_game.py --mode llm --model deepseek --gm smart --puzzle turtle_001
# 产出 logs/<model>_<puzzle>_<ts>.json
```

### 4.2 聚合评测
```bash
python run/evaluate.py
# 读 logs/ -> 产出：
#   analysis/metrics.csv   跨模型聚合（均值还原度/汤底保真/效率/轮次/声明率）
#   analysis/report.txt    文字结论（按还原度排序）
#   analysis/restoration.png  中文柱状图（自动注册 Noto CJK，无字体退英文标签）
```

### 4.3 报告怎么读
- `metrics.csv`：每行一个模型，列含 `restoration_mean` / `guess_score_mean` / `efficiency_mean` / `turns_mean` / `declared_rate`——横向比模型强弱。
- `restoration.png`：分组柱状图，一眼看出谁还原度高、谁效率高。
- `report.txt`：中文排序结论，直接摘抄发小红书/比赛。

### 4.4 指标聚合函数
`eval/metrics.py` 提供：
- `guess_score(declared, bottom)`：汤底 bigram 重合；
- `load_logs()` / `summarize()` / `aggregate_by_model()`：读取与跨模型聚合；
- `make_chart()` / `text_report()`：图表与文字结论。

---

## 5. 反模式（指标误用）

- ❌ 看到真实模型 `restoration` 低就判「模型差」→ 先查 GM 容错（SmartGM 是否配了 `keywords`）。
- ❌ 只看 `guess_score` 忽略 `restoration` → 蒙对也会被误判强。
- ❌ 单模型当 Judge 且只用一个 → 引入偏见；应多模型交叉。
- ❌ 改判定逻辑前没用数据证明必要性 → 违反「先度量后优化」。
