"""M1 冒烟测试：用假模型注入跑完整局，验证 LLMQuestioner 整条链路（无需 API key）。

验证目标：
1. LLMQuestioner 能构造提示词、调用 completion_fn、解析 JSON 动作协议。
2. 解析器能容忍模型常见的 ```json``` 代码围栏包裹。
3. 与 MockGM + Game 串联，能产出符合 specs/engine-design.md §7 的日志。
运行：/root/venv_aiarena/bin/python tests/test_m1_smoke.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import Game
from engine.puzzles import get_puzzle
from players.mock_player import MockGM
from players.llm_player import LLMQuestioner


# ---- 假模型：前两轮含关键事实的是非题，第三轮声明汤底 ----
# 第二轮故意用 ```json``` 围栏包裹，验证解析健壮性。
FAKE_RESPONSES = [
    '{"action": "ask", "content": "男人打嗝了吗？"}',
    '```json\n{"action": "ask", "content": "酒保用枪吓他了吗？"}\n```',
    '{"action": "guess", "content": "男人打嗝想喝水，酒保开枪吓好打嗝，他道谢离开。"}',
]


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


def fake_completion(model, messages, temperature=0.7, **kwargs):
    # 按调用次数顺序返回预设响应；用闭包计数器而非消息内容推断轮次。
    fake_completion.idx = getattr(fake_completion, "idx", 0)
    idx = min(fake_completion.idx, len(FAKE_RESPONSES) - 1)
    fake_completion.idx += 1
    return _Resp(FAKE_RESPONSES[idx])


def main():
    puzzle = get_puzzle("turtle_001")
    gm = MockGM()
    q = LLMQuestioner(model="deepseek/deepseek-chat", completion_fn=fake_completion,
                      max_questions=15)
    game = Game(puzzle, gm, q, max_turns=20, log_dir=tempfile.mkdtemp())
    log = game.run()
    path = game.save()

    # ---- 断言结构符合 §7 ----
    assert "meta" in log and "surface" in log and "bottom" in log
    assert log["meta"]["model"] == "deepseek/deepseek-chat"
    assert log["meta"]["puzzle_id"] == "turtle_001"
    assert "turns" in log and "result" in log
    assert log["result"]["declared"] is True
    # turtle_001 关键事实含 "男人打嗝" / "酒保用枪吓他" → MockGM 应回 yes → 命中 2/4
    assert log["result"]["restoration"] == 0.5, log["result"]
    assert set(log["result"]["key_facts_hit"]) == {"男人打嗝", "酒保用枪吓他"}

    with open(path, encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded["result"]["restoration"] == 0.5

    print("✅ M1 冒烟测试通过（假模型注入，无需 key）")
    print(f"   模型 : {log['meta']['model']}")
    print(f"   题目 : {log['meta']['puzzle_id']}")
    print(f"   轮数 : {log['result']['turns_used']}")
    print(f"   还原度: {log['result']['restoration']}（含围栏响应已正确解析）")
    print(f"   命中 : {log['result']['key_facts_hit']}")
    print(f"   日志 : {path}")


if __name__ == "__main__":
    main()
