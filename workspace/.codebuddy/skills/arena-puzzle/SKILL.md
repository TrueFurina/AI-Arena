---
name: arena-puzzle
description: "新增或校验 AI-Arena 题库题目（engine/puzzles.py 的字段：surface/bottom/key_facts/clue_gradient/keywords/clue_keywords），含 SmartGM 关键词容错格式。关键词：加题、出题、题库、puzzle、turtle、新题目"
version: "1.0.0"
author: "AI-Arena"
created: "2026-07-10"
updated: "2026-07-10"
---

# arena-puzzle · 题库出题

> **Goal**: 在 `engine/puzzles.py` 按规范新增海龟汤题目，确保 SmartGM 能正确判定还原度。

## When to Use

- 用户说「加一道题」「出新题库」「turtle_009 怎么写」「题库怎么扩展」
- 想扩大评测覆盖面（当前 8 题，目标 20+）

## 题目结构（每题一个 dict，加入 PUZZLES 列表）
```python
{
  "id": "turtle_009",
  "surface": "汤面：一句话情境（提问者可见，不泄底）",
  "bottom": "汤底：完整真相（GM 私有）",
  "key_facts": ["事实1", "事实2"],           # 还原度骨架，字符串列表
  "clue_gradient": ["相关但不核心的线索"],     # 部分相关信息
  "keywords": {                               # M2：事实 -> 触发词（容错改写）
      "事实1": ["词A", "词B"],
      "事实2": ["词C"],
  },
  "clue_keywords": {                          # M2：梯度线索 -> 触发词
      "相关但不核心的线索": ["词D"],
  },
}
```

## 字段规则
- `key_facts` / `clue_gradient`：**字符串列表**，向后兼容旧日志与指标；
- `keywords` / `clue_keywords`：**dict，键必须出现在上面的列表里**，值是触发词列表；
- SmartGM（`players/gm.py`）判定：
  - 提问含某 `key_facts` 的任一 `keywords` → 答 `yes`；
  - 含某 `clue_gradient` 的任一 `clue_keywords` → 答 `yes_and_no`；
  - 否则 `irrelevant`；
- 没配 `keywords` 时回退用事实原文本身（可用但容错差）。

## 出题 checklist
- [ ] `surface` 只给情境、不泄底；
- [ ] `bottom` 能由 `key_facts` 拼回；
- [ ] 每个 `key_facts` 配 1–3 个 `keywords`，覆盖模型的**常见自由表述**（如「男人打嗝」→ `["打嗝","打咯"]`）；
- [ ] `clue_gradient` 放相关但不核心的线索；
- [ ] 跑一次验证：`python run/single_game.py --mode mock --gm smart --puzzle turtle_009`，看还原度是否合理。

## 注意事项
- 取题入口是 `engine/puzzles.get_puzzle(id)`，新增后 `get_puzzle("turtle_009")` 可取；
- 改字段结构要保持向后兼容（旧日志聚合不崩）；
- 详见 `doc/EXTENDING.md` §2。
