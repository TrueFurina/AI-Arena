# P3 设计文档 · 贯通与报告（一键流水线 + 前端验签入口）

> 状态：**设计定稿 2026-10-13，实现未开始**。对应 ROADMAP P3（P3-1 一键流水线 /
> P3-2 前端凭证校验入口 / P3-3 对接出报告能力）。本文只定模块边界 / 数据流 /
> 失败模式 / 预注册验收，不含实现。所有「现状」均为仓库事实或实测数字。
>
> 执行纪律：口径单一真值、诚实负向披露、test/impl 分离提交、变异验证。
> 跑评测放量需在服务器 172.16.222.99；本阶段属流水线与前端，本机开发即可。

---

## 1. 目标与边界

**阶段目标**：把 P1（凭证）与 P2-A（多裁判共识）从「各自能跑的模块」变成
「一条能一键跑完并交付可信产物的链路」，并让第三方能独立验证产物。

不做（防范围蔓延，每条有理由）：

| 不做 | 理由 |
|---|---|
| 真实 LLM 裁判接入 | 属 P2-A A-7，**付费 API 阻塞**；本阶段面板仍用 `judges/biased.py` 受控模型，边界随产物声明 |
| 题库扩容 | 有实测依据的降级项（12 题足以支撑协议验证），非本阶段瓶颈 |
| 新指标 | 指标口径只来自 `engine/evaluate.py`，本阶段**不新增任何指标** |
| 前端重构 | 只在既有 `web_app.py` 上增 `/verify` 路由与页面，不动既有评测页 |
| 公链锚定 | 本地共识链，非公链；本阶段不宣称公级抗篡改（沿用 P1 边界） |

## 2. 三条关键设计判断（本阶段的价值全在这里）

### 判断 1：共识「拒绝发布」不得被流水线填补 —— 内核守卫

P2-A 实测出的**内核价值**是：共识面板在可靠权重不足时**拒绝发布结论**
（`score=None`），而单点裁判永远会给一个数字。若在流水线里写
`score = consensus.score or fallback_single_score`，就把内核当场退化成装饰——
与 P1 批评过的「盖戳式存证」同类错误，只是在更隐蔽的位置。

**设计**：`accepted=False` 时报告写 `REJECTED`，共识分数位置留空并注明
「未发布」，客观指标（F1 等）照报（它们不依赖裁判）。**任何路径都不得用
单点/等权分数填补共识位**。固化为测试断言（含变异验证）。

### 判断 2：共识结论必须进**签名对象**，否则不是「链上共识」

若共识结论只出现在 markdown 报告里，它与凭证无关——改报告即可改数字，
签名保护不到它。那就又退化成「对局存证 + 旁边另附一段共识文字」。

**设计**：把 `ConsensusVerdict.to_dict()` 写进 `EvaluationRecord.verdict`
（即被 ECDSA 签名的对象），与客观指标同等受完整性保护。
扩展走 `MatchRecorder.build_record(..., verdict_extra=...)`：
chain 层**不 import judges**（避免层间耦合），由流水线把共识字典喂进去。

### 判断 3：tx_id **不能**凭空验签 —— 必须写进 UI 文案

P1 凭证是**自包含**的（自带记录 + 公钥 + 签名）。验签需要凭证内容本身；
`tx_id` 只是链上交易索引，凭一个 tx_id 号无法验任何东西。
若前端做成「输入 tx_id → 显示✅通过」，那是伪验证。

**设计**：两个入口，行为不同且如实标注：
- **粘贴凭证 JSON** → 完整离线验签（digest / signature / 可选 identity 三步）
- **输入 tx_id** → 在**本地凭证库**（`logs/credentials/`）检索；命中则对
  该凭证做同样的完整验签；未命中如实报「本地凭证库中无此 tx_id」，
  **不得**显示为「验签失败」（两者语义不同，混淆即虚假陈述）

## 3. 模块边界与数据流

| 文件 | 改动 | 纪律符合点 |
|---|---|---|
| `chain/recorder.py` | `build_record` 增可选 `verdict_extra`；`None` 时行为**逐位不变** | 向后兼容有测试；chain 不依赖 judges |
| `pipeline.py`（新） | 编排 + CLI：`run_pipeline()` → `PipelineResult` | 指标只从 `engine/evaluate.py` 取 |
| `pipeline_report.py`（新） | 报告渲染（markdown），与编排分离便于测试 | 报告不含任何自行计算的指标 |
| `web_app.py` | 增 `/verify`（页面）+ `/api/verify`（POST：凭证 JSON 或 tx_id） | chain/flask 软导入，导入失败不拖垮既有评测页 |
| `tests/test_pipeline.py`（新） | §5 判据 + 向后兼容 + 三类失败区分 | 关键断言配变异验证 |

数据流：

```
pipeline.py  --puzzle X --questioner mock-gpt4o [--panel homo|hetero]
  ├─ S1 对局   TurtleSoupGame(...).run() → GameResult(dict)
  │            → logs/pipeline_game_<id>.json
  ├─ S2 指标   metrics_from_result(res)            # engine/evaluate.py 唯一真值
  ├─ S3 共识（可选，默认关）
  │     标定集：与被测题 disjoint（沿用 eval_judge_consensus 协议①），
  │             guess 用 solution 前缀构造质量梯度（协议③：不得取整段 solution）
  │     JudgePanel.calibrate() → consensus_verdict()
  │     accepted=False ⇒ score=None，报告标 REJECTED（判断 1）
  ├─ S4 凭证   MatchRecorder.build_record(res, verdict_extra=共识字典|None)
  │            → issue() → verify_credential(cred, identity=reg)
  │            链不可用 → 降级本地签名，anchor=None，不阻塞（P1 既有纪律）
  │            → logs/credentials/pipeline_<id>.credential.json
  └─ S5 报告   logs/pipeline_report_<id>.md
               含：对局信息 / 客观指标表 / 共识段 / 凭证段（record_hash、
               tx_id、anchor status、key_custody、验签三步）/ **验签入口命令**
               / 能力边界声明（不可省略）
```

## 4. 失败模式（pre-declared）

1. **链不可用**：降级仅本地签名，报告标注 `anchor=None` 与降级原因。
   凭证仍可验签（完整性不受影响），但**不具链上记录**——照实写
2. **共识拒绝发布**（判断 1）：报告 `REJECTED`，不得填补
3. **未标定即跑共识**：`JudgePanel` 已强制抛错；流水线须在开启 `--panel`
   时保证标定集非空且 disjoint，否则明确报错而非静默跳过
4. **tx_id 未命中本地凭证库**：报「未命中」而非「验签失败」（判断 3）
5. **身份校验不可用**：`identity_status=unavailable` 时不参与 ok 判定，
   UI 显示「未做身份校验」，不得显示为通过（沿用 `chain/verify.py` 语义）
6. **flask/chain 导入失败**：既有评测页不受影响；`/verify` 显式报不可用，
   不做静默降级（静默 fallback 比报错危险——项目既有教训）

## 5. 预注册验收标准（先定后跑）

| # | 判据 | 通过条件 | 不过的处置 |
|---|---|---|---|
| F1 | 一键贯通：单命令跑完 S1–S5 并落盘 4 类产物（对局日志/凭证/报告/报告内含凭证 ID） | 退出码 0 且产物齐全 | 阶段未完成 |
| F2 | **共识拒绝不被填补**：构造权重不足场景 → 报告共识位为空且标 REJECTED，无单点分数混入 | 报告与 `PipelineResult` 双处均为 None | 实现错误，必改 |
| F3 | **共识结论受签名保护**：篡改凭证 record 内的共识字典 → 验签失败 | 变红 | 实现错误，必改 |
| F4 | 向后兼容：不启用共识时，凭证 record 结构与 P1 逐字段一致（仅新增 `mode` 键） | 与历史凭证结构比对一致 | 实现错误，必改 |
| F5 | 前端验签：① 粘贴凭证 JSON → 三步结果如实展示；② 篡改任一字节 → 显示不可信；③ 未知 tx_id → 显示「未命中」而非「失败」 | 三条全部符合 | 实现错误，必改 |
| F6 | 降级不阻塞：链不可用时流水线退出码仍为 0，产物标记降级 | 通过 | 实现错误，必改 |

**变异验证**（全绿 ≠ 有效，项目既有纪律）：F2 / F3 / F5② 三条必须故意改错
确认会变红，留痕于提交信息。

## 5-A. 本机实测结果（2026-10-13，实现后回填）

命令与退出码（venv：`ai-arena-proto`，Python 3.13）：

| 命令 | 实测输出 |
|---|---|
| `python pipeline.py` | 对局 `c8f396e7` f1=0.3399；凭证 `204d1656…` 锚定=in-pool；验签 ✅（digest+signature）；退出码 0 |
| `python pipeline.py --panel hetero` | 共识**已发布** score=30.5559 ｜ 等权 30.51 ｜ **R=1.437** ｜ 后端=**cw-pbft**（本机真实链走通，未降级） |
| `python pipeline.py --panel hetero --absent j0,j1` | 共识**拒绝发布** score=None ｜ 等权 28.71 ｜ R=1.4369；报告共识位为 `—`，退出码仍为 0（拒绝发布 ≠ 流水线失败） |
| `python pipeline.py --panel hetero --identity` | 验签 ✅ 三步全通过，identity=**ok**（签发者公钥已登记于进程内身份合约） |

验收判据逐条：F1 ✅（4 类产物齐 + 报告含凭证 ID 与验签入口）｜F2 ✅ ｜F3 ✅ ｜
F4 ✅ ｜F5 ✅（含 tx_id 未命中报「未命中」）｜F6 ✅（链路径不存在时退出码仍 0，
degraded 记录降级原因）。测试 **119/119 全绿**（P3 新增 17 例）。

**变异验证（六条，全部被杀死）**：

| # | 变异 | 结果 |
|---|---|---|
| M1 | 报告用等权分数填补「共识结论分数」位 | ✅ 变红（F2 有效） |
| M2 | `verdict_extra` 改 None（共识不进签名对象） | ✅ 变红（F3 有效，两条 f3 全红） |
| M3 | tx_id 未命中的 `ok` 改 False | ✅ 变红（F5③ 有效） |
| M4 | 链不可用降级不写 degraded | ✅ 变红（F6 有效） |
| M5 | 拒绝发布不写 degraded | ✅ 变红（F2 降级记录有效） |
| M6 | 忽略 `use_identity`（不传 identity 给验签） | ✅ 变红（身份路径未被静默跳过） |

**实现中修掉的两个坑（设计阶段未预见，留痕）**：
1. `MockJudge` 没有 `.id`（Judge 契约只保证 `.score`）→ 裁判标识改为
   `id → model → 规格串` 逐级回退，不伪造一个「看起来正规」的 id
2. `IdentityRegistry.available` 是 **property** 不是方法，写成 `reg.available()`
   直接 `TypeError: 'bool' object is not callable` → 改为兼容两种形态。
   同类教训第二次（属性 vs 方法），今后凡新接外部对象先确认其形态

## 6. 与其它阶段的关系

- **P1**：本阶段复用 `MatchRecorder` / `verify_credential`，不改其验签语义
- **P2-A**：本阶段把共识结论送进签名对象（判断 2），使 P2-A 的结论可被第三方验证
- **P2-B**：MARL 对局日志同样可走本流水线（对局产物格式一致），不依赖
- **P3-3（报告对接 `workspace/run/report_share.py`）**：本阶段先产本项目自有的
  markdown 报告；资产库脚本对接列为 P3 末项，不阻塞 F1–F6

## 7. 诚实边界（随产物声明，不得省略）

1. 面板裁判是**测量误差模型**（`judges/biased.py`），不是真实 LLM 裁判；
   受控结论不可外推为真实裁判的聚合增益
2. 本地共识链 ≠ 公链；`anchor=in-pool` 不具最终性
3. 验签通过 **≠** 凭证由本评测台签发，除非接了身份锚定且该 id 受控登记
4. tx_id 检索依赖本地凭证库，不是「任何人输入任意 tx_id 都能验」
