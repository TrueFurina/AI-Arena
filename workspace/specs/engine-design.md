# 引擎技术规格（Engine Design Spec）

> 本文是 `engine/game.py` 与 `players/*` 的**实现契约**。先定协议再写代码，避免边写边改。
> 对应规划见 `doc/DESIGN.md`、`doc/ARCHITECTURE.md`、`doc/GAME.md`。

## 1. 范围与目标（M0–M4）

- 首发实现**模式 A：单 GM + 单提问者**（GM 守汤底，提问者用是非题还原）。
- 支持 `mock` 模式：GM 与提问者都可用规则化假实现，零 key 跑通整条 pipeline。
- 真实模型通过 litellm 接入（见 `players/llm_player.py`）。
- 每局产出一份结构化 JSON 日志，供 `eval/` 评测。

## 2. 游戏状态机

```
        ┌─────────┐
        │  INIT   │  初始化：载入题库、建角色、注入汤面
        └────┬────┘
             ▼
        ┌─────────┐
        │ RUNNING │  回合循环：questioner.act(obs) → gm.answer(q) → 记录
        └────┬────┘
   ┌─────────┼───────────────┐
   │ turn>=max_turns         │ questioner 主动 guess（声明汤底）
   │ 或 guess 被接受          │
   ▼                         ▼
        ┌─────────┐     ┌──────────────┐
        │ FINISHED│ ←───│ 判定还原度    │
        └────┬────┘     └──────────────┘
             ▼
        输出 logs/*.json
```

- `max_turns` 默认 20（可由 `.env` 的 `ARENA_MAX_TURNS` 覆盖）。
- 状态由 `Game` 类持有：`phase ∈ {init, running, finished}`、`turn`、`history`、`revealed`。

## 3. 回合协议

每一回合：

1. 引擎构造 `observation`（见 §5）传给当前提问者。
2. 提问者调用 `Player.act(observation)`，返回动作 `{"action":"ask"|"guess", ...}`。
3. 若为 `ask`：GM 生成是非回答 `{"answer": "yes"|"no"|"irrelevant"|"yes_and_no"|"unknown"}`，追加进 `history`。
4. 若为 `guess`：进入判定，比对声明汤底与真实汤底。
5. `turn += 1`；若 `turn >= max_turns` 或已 guess，进入 `FINISHED`。

## 4. 角色契约（roles.py）

| 角色 | 输入 | 输出 | 目标 |
|---|---|---|---|
| **GM** | 当前已揭示线索 + 玩家提问 | 是非类限定回答 | 严守汤底，引导但不泄露 |
| **提问者** | 汤面 + 历史问答 | 一个是非题 / 一次 guess | 用最少步数还原汤底 |

> 模式 C（骗子，M5）与模式 B/D 在此规格后续版本扩展，先预留 `Role` 枚举与 `Player` 多态接口。

## 5. JSON 动作协议（核心，必须严格遵守）

### 5.1 传给提问者的 observation
```json
{
  "surface": "汤面文本",
  "history": [
    {"role": "questioner", "content": "男人打嗝了吗？"},
    {"role": "gm", "answer": "yes"},
    {"role": "questioner", "content": "他想喝水？"},
    {"role": "gm", "answer": "yes"}
  ],
  "turn": 3,
  "max_turns": 20
}
```

### 5.2 提问者输出（Player.act 返回值）
```json
// 继续提问
{"action": "ask", "content": "酒保用枪吓了他一跳？"}
// 主动声明汤底（结束本局）
{"action": "guess", "content": "汤底：男人打嗝想喝水压惊，酒保开枪吓好打嗝，他道谢离开。"}
```

### 5.3 GM 输出（回答）
```json
{"answer": "yes"}          // 是
{"answer": "no"}           // 不是
{"answer": "irrelevant"}   // 不重要 / 与真相无关
{"answer": "yes_and_no"}   // 是也不是
{"answer": "unknown"}      // 无法判断
```

> 所有输出由 `pydantic` 模型校验，解析失败视为该玩家本轮"空过"或触发重试（由 `Game` 决定）。

## 6. 题库数据结构（puzzles.py）

```json
{
  "id": "turtle_001",
  "surface": "一个人走进酒吧……",
  "bottom": "男人打嗝……完整真相",
  "key_facts": ["男人打嗝", "想喝水压惊", "酒保用枪吓他", "打嗝被治好"],
  "clue_gradient": ["与饮水有关", "与声音有关", "与惊吓有关"]
}
```

- `key_facts` 用于计算**还原准确率**。
- `clue_gradient` 用于信息增益的近似估算（M2 起）。

## 7. 对局日志格式（logs/<model>_<puzzle>_<ts>.json）

```json
{
  "meta": {
    "model": "mock / deepseek/deepseek-chat",
    "puzzle_id": "turtle_001",
    "mode": "single",
    "max_turns": 20,
    "timestamp": "2025-07-10T19:40:00"
  },
  "surface": "汤面文本",
  "bottom": "真实汤底",
  "key_facts": ["..."],
  "turns": [
    {"turn": 1, "question": "男人打嗝了吗？", "answer": "yes"},
    {"turn": 2, "question": "他想喝水？", "answer": "yes"}
  ],
  "result": {
    "declared": true,
    "restoration": 0.75,
    "guess_score": 0.60,
    "efficiency": 0.062,
    "turns_used": 12,
    "key_facts_hit": ["男人打嗝", "想喝水压惊", "打嗝被治好"]
  }
}
```

- `restoration` = 命中的 key_facts / 总 key_facts。**M2 起为关键词感知**：命中判定用 `puzzle.keywords` 做关键词重叠，而非要求提问逐字包含事实原文（配合 `SmartGM`）。
- `guess_score` = 声明汤底与真实汤底的二元组 Jaccard 重合度（0~1），衡量"最终猜到了多少真相"。
- `efficiency` = `restoration / turns_used`，单位轮次的还原效率。
- `eval/metrics.py` 直接读此结构计算推理指标（见 §8 `eval/`）。

## 8. 目录结构与文件职责（M0 落地）

```
AI-Arena/
├── .env.example              ✅ 已建
├── requirements.txt          ✅ 已建
├── engine/
│   ├── game.py               ⏳ 建：Game 状态机 + 回合循环 + 日志输出
│   ├── puzzles.py            ⏳ 建：题库（≥3 条示例）
│   └── roles.py              ⏳ 建：Role 枚举 + 角色契约
├── players/
│   ├── base.py               ✅ Player/GM 抽象（act / answer 契约）
│   ├── llm_player.py         ✅ litellm 适配（M1 落地，含别名/重试/解析容错）
│   ├── mock_player.py        ✅ 规则化假玩家（M0 先跑通）
│   └── gm.py                 ✅ SmartGM 关键词感知主持人（M2，零 key）
├── prompts/                  ✅ 提问者 system prompt（questioner.md）
├── run/
│   ├── single_game.py        ✅ 编排一局（mock / llm / --gm / --dry-run）
│   └── evaluate.py           ✅ 评测入口（M2，产出 csv/txt/png）
├── tests/                    ✅ 冒烟测试（test_m1_smoke / test_m2_smoke，无需 key）
├── eval/
│   └── metrics.py            ✅ 推理指标：还原度/汤底保真/效率 + 聚合 + 图表 + 结论
├── eval/
│   ├── metrics.py            ⏳ 建：读日志算指标
│   └── report.py             ⏳ 建：出图 + 文字结论
├── logs/                     ⏳ 建：对局 JSON
├── analysis/                 ⏳ 建：评测报告输出
├── doc/                      ✅ 6 篇规划文档
└── specs/engine-design.md    ✅ 本文件
```

## 9. 验收（M0 完成标准）

1. `python run/single_game.py --mode mock` 能跑完一局并产出 `logs/*.json`。
2. 日志字段符合 §7 结构，且 `restoration` 等字段可被 `eval/metrics.py` 解析。
3. 全程不依赖任何 API key。
