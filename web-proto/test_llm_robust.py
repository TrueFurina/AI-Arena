"""players/llm.py 容错测试：验证重试 + 按角色降级，对局不因单次调用失败而崩。

不依赖真实网络（mock 掉 litellm.completion），用 /tmp/venv 的 python 跑：
    /tmp/venv/bin/python test_llm_robust.py
"""
import sys
import types

sys.path.insert(0, ".")

import litellm

call_log = []


class _Msg:
    def __init__(self, c):
        self.content = c


class _Resp:
    def __init__(self, c):
        self.choices = [types.SimpleNamespace(message=_Msg(c))]


def _make_fake(behaviors):
    """behaviors: 依次模拟 'empty' / 'err' / 'ok:文本'。"""
    state = {"i": 0}

    def fake(*a, **k):
        b = behaviors[min(state["i"], len(behaviors) - 1)]
        state["i"] += 1
        call_log.append(b)
        if b == "empty":
            return _Resp("")
        if b == "err":
            raise RuntimeError("simulated API error")
        return _Resp(b.split(":", 1)[1])

    return fake


from players import llm as L
from engine.game import Puzzle, TurtleSoupGame

pz = Puzzle("p1", "t", "一个人走进餐厅点了一碗海龟汤，喝了一口就哭了。", "他曾在海难中靠同伴的肉活命，汤让他认出那是同伴。")

# TEST1：空->空->成功，应重试并返回成功内容
call_log.clear()
litellm.completion = _make_fake(["empty", "empty", "ok:你好"])
out = L._call("zai/glm-4-flash", "sys", [{"role": "user", "content": "hi"}], max_tokens=10)
print(f"TEST1 重试后成功: {out!r} | 调用次数={len(call_log)} (期望3)")

# TEST2：全部失败应抛 LLMError
call_log.clear()
litellm.completion = _make_fake(["err", "err", "err"])
try:
    L._call("zai/glm-4-flash", "sys", [{"role": "user", "content": "hi"}], retries=2)
    print("TEST2 FAIL: 未抛异常")
except L.LLMError as e:
    print(f"TEST2 抛 LLMError OK: {str(e)[:50]} | 调用次数={len(call_log)} (期望2)")

# TEST3：角色降级 —— host 失败应回 '不重要'
call_log.clear()
litellm.completion = _make_fake(["err", "err", "err"])
host = L.LLMHost(model="zai/glm-4-flash")
print(f"TEST3 host 降级: {host.answer(pz, '问题?', [])!r} (期望 '不重要')")

# TEST4：judge 失败应回 (0.0, '（裁判无响应，未评分）')
litellm.completion = _make_fake(["err", "err", "err"])
judge = L.LLMJudge(model="zai/glm-4-flash")
print(f"TEST4 judge 降级: {judge.score(pz, '还原文本')!r}")

# TEST5：questioner 失败应回兜底问题（含 '模型无响应'）
q = L.LLMQuestioner(id="q1", model="zai/glm-4-flash")
litellm.completion = _make_fake(["err", "err", "err"])
print(f"TEST5 questioner 降级: {q.ask_question(pz, [])[:30]!r}")

print("--- 容错逻辑全部验证完毕 ---")
