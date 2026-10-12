# M3 多模型对战 · 设计文档

> 本文定义 M3（多模型对战 + 自动评测）的工程设计与约定，配套实现见 `run/tournament.py` 与 `config.toml`。

---

## 1. 目标

- 用一个命令，对**注册表里的多个模型** × **每题多局**自动跑完，产出可横向比较的评测数据。
- **零 key 即可验证编排**：mock 基线（strong/weak）立即可跑，真实模型填 key 后自动接入。
- 跑完自动调用 `evaluate.py` 聚合出 `analysis/` 报告，闭环「对局 → 日志 → 指标 → 报告」。

---

## 2. 模型注册表（`config.toml`）

用 TOML 描述参与对战的模型，避免把模型清单写死在代码里。

```toml
[globals]
max_turns = 20
runs_per_puzzle = 2
gm = "smart"
log_dir = "logs"
evaluate_after = true

[[models]]
alias = "deepseek"
litellm = "deepseek/deepseek-chat"
enabled = false
note = "DeepSeek，推理强、便宜"

[[models]]
alias = "mock-strong"
mock = "strong"
enabled = true
```

### 字段语义
| 字段 | 位置 | 含义 |
|---|---|---|
| `alias` | 每条 | 展示名（日志 `meta.model` 与报告分组用） |
| `litellm` | 每条 | 完整 litellm 模型名；**留空 = 用 mock 玩家** |
| `enabled` | 每条 | 是否参与；真实模型需在 `.env` 填 key 后才置 true |
| `mock` | 每条 | `"strong"` 逐事实问（还原度≈1）/ `"weak"` 问无关（还原度≈0） |
| `max_turns` | globals | 单局最大轮数 |
| `runs_per_puzzle` | globals | 每题每模型跑几局（取均值更稳定） |
| `gm` | globals | `smart`(关键词感知) / `mock`(逐字) |
| `log_dir` | globals | 日志目录 |
| `evaluate_after` | globals | 跑完是否自动聚合报告 |

> Python 3.11+ 用标准库 `tomllib` 解析，无额外依赖。

---

## 3. 循环逻辑（`run/tournament.py`）

```
for m in models if enabled:
    for p in puzzles:                      # 全部题，或 --puzzle 指定
        for r in range(runs_per_puzzle):   # 每题多局
            q = build_questioner(m, p)     # mock / LLM 按条目构造，每局新建（状态隔离）
            game = Game(p, gm, q, max_turns, log_dir)
            log = game.run(); game.save()  # 落盘 logs/<model>_<puzzle>_<ts>.json
if evaluate_after:
    subprocess.run(["python","run/evaluate.py"])   # 聚合出 analysis/
```

### 提问者构造
- `mock="strong"` → `MockQuestioner(facts, bottom)`：逐事实提问，SmartGM 下还原度≈1.0；
- `mock="weak"` → `WeakMockQuestioner(facts, bottom)`：问无关问题，还原度≈0.0；
- 真实模型 → `LLMQuestioner.from_model(litellm)`，靠 `.env` 的 key 自动接入；
- `--mock` 开关：把所有真实模型临时替换为 `mock-strong`，纯粹验证零 key 编排。

---

## 4. 输出与评测闭环

- 每局 → `logs/<model>_<puzzle>_<ts>.json`（结构见 `specs/engine-design.md` §7）；
- `evaluate.py` 读 `logs/` → `analysis/metrics.csv` + `report.txt` + `restoration.png`；
- `metrics.csv` 每行一局，列含 `restoration`/`guess_score`/`efficiency`/`turns_used`/`declared`；
- 跨模型聚合（`eval/metrics.aggregate_by_model`）按 `model` 聚均值，报告按还原度排序。

> 注：`evaluate.py` 聚合 `logs/` 下**所有**日志。要限定某次对战范围，可指定独立 `log_dir` 或先清空旧日志。

---

## 5. 零 key 验证策略

不依赖任何 key 即可证明链路正确：
1. `python run/tournament.py --mock` → 真实模型也用 mock 跑，验证编排不崩；
2. 预期：`mock-strong` 还原度≈1.0、`mock-weak`≈0.0（指标区分度正确）；
3. 真实 key 到位后，把对应 `[[models]]` 的 `enabled` 改 true、填 `.env`，再跑即可。

---

## 6. 演进（M5/M6）

- **M5 骗子变体**：模型条目加 `role = "liar"|"truthful"`，`Game` 按角色决定 GM 真伪分支；指标加「误导成功率/被识破率」。
- **M6 辩论 + LLM-as-Judge**：在 `Game` 结束后加 `debate` 阶段，引入 `players/judge.py`（多模型交叉裁判）；注册表加 `judge` 段。
- **更多玩法**：`[[models]]` 可扩展为「玩家组合」而非单模型，支撑双盲互猜（玩法 B）。

---

## 7. CLI 速查

```bash
python run/tournament.py                  # 按 config.toml 跑全部 enabled
python run/tournament.py --mock          # 零 key 验证编排
python run/tournament.py --model deepseek --puzzle turtle_001 --runs 2
python run/tournament.py --no-evaluate    # 不自动聚合
python run/tournament.py --config my.toml # 指定注册表
```

详见 `.codebuddy/skills/arena-tournament/SKILL.md`。
