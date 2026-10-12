---
name: arena-deceiver
description: "M5 骗子变体：让 GM 按比例撒谎（DeceiverGM），度量提问者在不可信信息下的推理鲁棒性。零 key 可运行演示。关键词：骗子、欺骗、撒谎、deceiver、鲁棒性、robustness、M5、对抗"
version: "1.0.0"
author: "AI-Arena"
created: "2026-07-17"
updated: "2026-07-17"
---

# arena-deceiver · 骗子变体（欺骗维度）

> **Goal**: 用 `run/deceiver_demo.py` 演示 M5 —— 同一提问者在「诚实 GM」与「撒谎 GM」下跑同一套题，度量**欺骗鲁棒性**。

## When to Use

- 用户说「加撒谎」「测试模型抗骗能力」「骗子变体」「鲁棒性怎么测」
- 想验证「不可信信息」下模型的推理是否崩溃
- 要扩展指标到 deception 维度（lie_rate / robustness）

## 用法

### 零 key 演示
```bash
python run/deceiver_demo.py                 # 默认 lie_rate=0.3, seed=42
python run/deceiver_demo.py --lie-rate 0.5 --seed 7
# 输出每题：诚实还原度 | 撒谎还原度 | 鲁棒性 | 实际撒谎率
```

### 在代码里用 DeceiverGM
```python
from players.gm import DeceiverGM
from engine.game import Game
from players.mock_player import MockQuestioner

gm = DeceiverGM(lie_rate=0.3, rng=__import__("random").Random(42))  # 固定种子可复现
q  = MockQuestioner(key_facts=p["key_facts"], bottom=p["bottom"])
log = Game(p, gm, q).run()
print(log["result"]["deception"])   # {'lies_total': 3, 'lie_rate': 0.75}
```

## 指标（详见 specs/deceiver-design.md）
| 指标 | 含义 |
|---|---|
| `lie_rate` | 实际撒谎轮数 / 已回答轮数 |
| `deception_robustness` | 撒谎环境还原度 / 诚实环境还原度 ∈ [0,1]；越高越抗骗 |

## 实现要点
- `DeceiverGM` 继承 `SmartGM`，**只翻转 yes/no**（irrelevant/yes_and_no/unknown 不撒谎）。
- `lied` 字段写进历史与日志，**不进观察**（提问者看不到被骗，否则指标失真）。
- `engine/game.py` 向后兼容：旧日志无 `lied` 时记 0。

## 注意事项
- 当前 PoC 是 V1（Deceiver GM）。V2（提问者主动 challenge 检测谎言）与 V3（卧底）见设计规格，尚未实现。
- 鲁棒性必须「同提问者跨诚实/撒谎」对比，别用单局绝对值（会混入模型本身强弱）。
- 真实 key 就绪前，先用 mock 验证引擎层通畅，不要过早投资 V2/V3（先度量后优化）。
