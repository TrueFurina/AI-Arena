# AI-Arena · 项目愿景与四维度蓝图

> 一份给「接手者 / 协作者 / 比赛评委」看的北极星文档。它回答三件事：
> **我们做什么、为什么这么做、下一步往哪走。**
> 工程纪律见 `doc/METHODOLOGY.md`；里程碑状态见 `ROADMAP.md`；指标数学见 `EVALUATION.md`。

---

## 1. 一句话定位

**AI-Arena 是一个离线、零人工干预的 AI-vs-AI 社交推理评测平台**——
让大模型以结构化协议互相出题、提问、辩论，用「最少轮次内还原多少关键事实」
这类**可复现、可审计**的指标，量化模型的提问与推理能力。

起点是海龟汤（情境推理谜题），但它只是载体；真正的目标是**一套可扩展的多维推理评测方法论**。

---

## 2. 设计哲学（为什么这样建）

| 原则 | 在我们项目里的体现 |
|------|-------------------|
| **零 key 优先** | 任何新功能先有零 key 可跑的 mock / dry-run 验证，再接真实模型 |
| **先度量后优化** | 所有指标来自真实落盘日志，不靠主观印象；加功能前先能测 |
| **配置 / 代码分离** | 模型注册表 `config.toml`、key 只在 `.env`，代码不写死 |
| **可复现可审计** | 结构化 JSON 动作协议 + 关键词感知 GM，指标可被任何人复算 |
| **渐进式接入** | mock → dry-run → 单模型 → 多模型 → 变体，每个里程碑独立可交付 |
| **知识沉淀** | 大版本更新 `MEMORY.md`（规则 3），技能自动化（`doc/` + `.codebuddy/skills`） |

---

## 3. 四维度演进蓝图

我们把「社交推理能力」拆成四个可独立度量、又可组合的维度。
每个维度都有独立的游戏模式、指标与里程碑。

```
        A. 推理（单人探测真相）
        │   海龟汤：提问者 vs GM，用最少是非题还原关键事实
        │   指标：restoration / guess_score / efficiency
        ▼
B. 双盲互猜（角色对称）
│   双方互为 GM+提问者，互探对方汤底
│   指标：互探还原度差 / 收敛轮次
▼
C. 骗子变体（信息不可信）           D. 辩论 + Judge（立场对抗）
│   GM 按比例撒谎，测鲁棒性          │   正反方多轮对抗，LLM 裁判打分
│   指标：deception_robustness       │   指标：persuasion_margin / consensus
▼                                   ▼
        （对抗层可叠加到 A/B：撒谎的互猜、被裁判裁决的辩论）
```

| 维度 | 一句话 | 核心指标 | 里程碑 | 状态 |
|------|--------|----------|--------|------|
| **A 推理** | 单人用最少问题探测真相 | restoration / guess_score / efficiency | M0–M3 | ✅ 已落地（零 key 可跑，真实 key 待填） |
| **B 双盲互猜** | 双方互探汤底，角色对称 | 互探还原度差 / 收敛轮次 | M4+ | 📋 规划（见 `EXTENDING.md` 模式 B） |
| **C 骗子变体** | 信息可能不可信，测抗骗 | deception_robustness / lie_detection | M5 | 🟡 PoC 已落地（V1 Deceiver GM） |
| **D 辩论 + Judge** | 立场对抗 + 自动裁判 | persuasion_margin / consensus | M6 | 🟡 PoC 已落地（框架 + Mock 裁判） |

> 维度不是孤岛：**对抗层可叠加**——可以跑「撒谎的互猜」（C×B）、「被裁判裁决的辩论」（D 独立或 D×A）。
> 这让评测从「单点能力」长成「能力谱」。

---

## 4. 里程碑总览（M0–M6）

| 里程碑 | 内容 | 真实 key 依赖 | 状态 |
|--------|------|---------------|------|
| M0 | mock 跑通整条 pipeline，结构化 JSON 日志 | 否 | ✅ |
| M1 | litellm 统一接入真实模型提问者（惰性导入，零 key 不阻塞） | 是（待填） | 🟡 代码就绪，key 阻塞 |
| M2 | SmartGM 关键词感知 + 推理指标 | 否 | ✅ |
| M3 | 多模型对战脚手架（config.toml 注册表 + tournament） | 否（mock 可跑） | ✅ 零 key 验证通过 |
| M4 | 可读战报（小红书 / 比赛友好，SHARE.md） | 否 | ✅ |
| M5 | 骗子变体（欺骗维度，DeceiverGM + robustness） | 否（PoC） | 🟡 PoC 落地 |
| M6 | 辩论 + LLM-as-Judge（辩论维度） | 否（PoC） | 🟡 PoC 落地 |

> **当前最大阻塞**：真实 LLM 对局被 key 阻塞（netdrive 附件故障 + tokenc.top 实测非 LLM 后端）。
> 所有代码/文档/脚手架已就位，拿到 key 即 `config.toml` 改 `enabled=true` 一键跑（见 `KEYS.md`）。

---

## 5. 对社区与比赛的定位

**可传播点（小红书 / 技术博客）**
- 「让 AI 互相出题玩海龟汤，谁的逻辑更强？」——天然有梗、可视化友好。
- 反直觉指标「还原度」：不是谁答得快，而是**谁用最少问题问出真相**。
- 零 key 可复现：任何人 `python run/tournament.py --mock` 就能看到战力榜。

**可投稿点（比赛 / 技术报告）**
- 问题定义清晰：离线、零人工干预、结构化协议驱动的多维推理评测。
- 方法可复现：关键词感知 GM 做可复算判定；litellm 统一接入；TOML 注册表 + 零 key 冒烟。
- 维度可扩展：A/B/C/D 四维度提供「能力谱」而非单点分数，避免过拟合单一指标。
- 诚实声明局限：真实模型对局尚未跑（key 阻塞），当前数据以 mock 基线为主。

---

## 6. 对外开放的接口约定

任何维度都遵循同一套「可组合」契约，方便外部贡献：

1. **玩家契约** `players/base.py`：`act(observation) -> {action, content}`（JSON 协议）。
2. **GM 契约**：`answer(question, context) -> {answer, [lied]}`。
3. **裁判契约**（M6）：`score(transcript) -> {winner, scores, rationale}`。
4. **日志契约**：每份对局落盘为 JSON，含 `meta` / `result`，指标模块纯离线复算。
5. **注册表契约** `config.toml`：`[[models]]` 声明 alias / litellm / enabled，零 key 与真实模型同 pipeline。

> 想加新维度？先写零 key 的 mock 实现 + 设计规格（放 `specs/`）+ 一个 `.codebuddy/skills/` 技能，
> 跑通后再接真实模型（见 `EXTENDING.md`）。
