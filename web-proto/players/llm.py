"""真实 LLM 玩家适配层：用 litellm 统一调用各家模型。

- 模型名格式：provider/model（见 API使用教程.md）
- Key 统一从 .env 读取（见 API_KEYS.md），代码不写死
- 对 zai/ qwen/ zhipuai/ 前缀自动把 system 合并进 user（国产模型兼容）
- 自定义端点 ashbwin 通过 ASHBWIN_API_KEY / ASHBWIN_API_BASE 接入
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

from .base import Player, Host, Judge
from engine.game import Puzzle

# provider 前缀 -> 环境变量名（与 web_app._env_key_for_model 保持一致）
_ENV_KEY_MAP = {
    "zai/": "ZAI_API_KEY",
    "zhipuai/": "ZAI_API_KEY",
    "moonshot/": "MOONSHOT_API_KEY",
    "deepseek/": "DEEPSEEK_API_KEY",
    "openai/": "OPENAI_API_KEY",
    "qwen/": "DASHSCOPE_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "gemini/": "GOOGLE_API_KEY",
}
# 需要合并 system -> user 的国产模型前缀
_MERGE_SYSTEM_PREFIXES = ("zai/", "zhipuai/", "qwen/")


def env_key_for_model(model: str) -> Optional[str]:
    if model == "ashbwin" or model.startswith("ashbwin/"):
        return "ASHBWIN_API_KEY"
    for prefix, env in _ENV_KEY_MAP.items():
        if model.startswith(prefix):
            return env
    return None


def _api_base_for(model: str) -> Optional[str]:
    if model == "ashbwin" or model.startswith("ashbwin/"):
        return os.getenv("ASHBWIN_API_BASE")
    return None


def _merge_system(prefixes, system: str, messages: List[dict]) -> List[dict]:
    """把 system 内容合并进第一条 user 消息（国产模型兼容）。"""
    if not system:
        return messages
    out = []
    merged = False
    for m in messages:
        if m["role"] == "user" and not merged:
            out.append({"role": "user", "content": f"{system}\n\n{m['content']}"})
            merged = True
        else:
            out.append(m)
    if not merged:
        out.insert(0, {"role": "user", "content": system})
    return out


class LLMError(RuntimeError):
    """真实模型调用在重试后仍失败，由调用方按角色降级处理。"""


def _call(
    model: str,
    system: str,
    conversation: List[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 400,
    retries: int = 3,
    timeout: int = 60,
    extra_body: Optional[dict] = None,
) -> str:
    """统一调用入口（懒加载 litellm，仅在用到真实模型时依赖）。

    对空响应与异常做重试（指数退避，尊重 429 限速）；全部失败后抛 LLMError，
    由各 Player 方法按角色给出降级回复，保证对局不会因单次调用失败而中断。

    extra_body：端点私有参数透传（默认 None = 行为与此前逐位一致）。
    存在的理由：glm-4.5 等 **thinking 模型**会把 max_tokens 全部消耗在
    reasoning 上（实测 finish_reason=length、content 为空、reasoning 1399
    字符），调用方看到的是 `empty_content`——需显式传
    `{"thinking": {"type": "disabled"}}` 才能拿到正文。
    """
    import litellm

    api_key = os.getenv(env_key_for_model(model) or "", "")
    api_base = _api_base_for(model)
    # 国产 OpenAI 兼容端点：把 zai/<m> / moonshot/<m> 改写为 openai/<m> + 自定义 base，
    # 这样无需每次 litellm.register_model 也能直接调用（ashbwin 仍走 arena_providers 注册）。
    litellm_model = model
    if model.startswith("zai/"):
        litellm_model = "openai/" + model[len("zai/"):]
        api_base = api_base or "https://open.bigmodel.cn/api/paas/v4"
    elif model.startswith("moonshot/"):
        litellm_model = "openai/" + model[len("moonshot/"):]
        api_base = api_base or "https://api.moonshot.cn/v1"
    msgs = list(conversation)
    if any(model.startswith(p) for p in _MERGE_SYSTEM_PREFIXES):
        msgs = _merge_system(_MERGE_SYSTEM_PREFIXES, system, msgs)
    else:
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
    kwargs = {
        "model": litellm_model,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "num_retries": 0,  # 重试逻辑自己控制，避免与 litellm 内置叠加
    }
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    if extra_body:
        kwargs["extra_body"] = extra_body

    last_err = "unknown"
    for attempt in range(1, retries + 1):
        try:
            resp = litellm.completion(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
            last_err = "empty_content"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
        if attempt < retries:
            time.sleep(min(2 ** attempt, 8))  # 指数退避，尊重 429 限速
    raise LLMError(f"model={model} retries={retries} last={last_err}")


def _transcript_to_msgs(transcript: List[dict], speaker_id: str) -> List[dict]:
    """把共享对局记录转成单模型可见的对话：问题=user，回答=assistant。

    注意：不带【】方括号标签。实测（2026-10-12，glm-4-flash）弱模型会把
    【主持回答】等标签原样抄进输出（格式污染），去掉标签后由 role 本身
    承载语义，避免诱导模仿。
    """
    msgs = []
    for m in transcript:
        if m["role"] == "system":
            continue
        if m["role"] == "question":
            msgs.append({"role": "user", "content": m["content"]})
        elif m["role"] == "answer":
            msgs.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "guess" and m["speaker"] == speaker_id:
            msgs.append({"role": "user", "content": f"【我的还原】{m['content']}"})
    return msgs


class LLMQuestioner(Player):
    def ask_question(self, puzzle: Puzzle, transcript: List[dict]) -> str:
        # 输出格式约束（2026-10-12 加固）：实测 glm-4-flash 在无约束时会输出
        # 陈述句/句号/元评论。加约束后有效提问率 2/8→7/16（4 局样本，方向明确
        # 但未达统计显著）。残留问题：R3 起（历史累积 4+ 条）仍高频退化，
        # 判定为免费模型固有能力上限，非 prompt 可根治——需要更强模型或
        # 缩短 max_rounds，见 diag_replay.py 可复现。
        sys = (
            "你是海龟汤（情境推理）游戏的提问者。你只能通过 yes/no 类问题还原真相。"
            "输出必须是一个以问号结尾的疑问句；即使信息不足，也要基于汤面做出猜测性提问，"
            "不要解释、不要寒暄、不要复述规则。"
        )
        conv = [{"role": "user", "content": f"汤面：{puzzle.surface}"}]
        conv += _transcript_to_msgs(transcript, self.id)
        try:
            return _call(self.model, sys, conv, max_tokens=120)
        except LLMError:
            return "（模型无响应）请告诉我，这件事发生在室内还是室外？"

    def final_guess(self, puzzle: Puzzle, transcript: List[dict]) -> str:
        sys = "你是海龟汤提问者。请基于全部问答，写出你对该事件完整真相的还原（一段连贯文字）。"
        conv = [{"role": "user", "content": f"汤面：{puzzle.surface}"}]
        conv += _transcript_to_msgs(transcript, self.id)
        conv.append({"role": "user", "content": "现在请写出你的最终还原。"})
        try:
            return _call(self.model, sys, conv, temperature=0.3, max_tokens=500)
        except LLMError:
            return "（模型无响应，无法还原）"


class LLMHost(Host):
    def __init__(self, model: str = "mock-host"):
        super().__init__(id="host", model=model)

    def answer(self, puzzle: Puzzle, question: str, transcript: List[dict]) -> str:
        sys = (
            f"你是海龟汤的主持人，掌握完整真相。\n"
            f"汤面：{puzzle.surface}\n"
            f"汤底：{puzzle.solution}\n"
            "规则：对玩家的 yes/no 问题，只能从以下选项作答：是 / 否 / 部分是 / 无关 / 不重要。"
            "严格依据汤底，不得泄露汤底原文或额外信息。"
        )
        conv = _transcript_to_msgs(transcript, self.id)
        conv.append({"role": "user", "content": f"玩家问：{question}"})
        try:
            raw = _call(self.model, sys, conv, temperature=0.2, max_tokens=40)
        except LLMError:
            return "不重要"
        # 归一化
        for kw in ("部分是", "不重要", "无关", "否", "是"):
            if kw in raw:
                return kw
        return raw


class LLMJudge(Judge):
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model

    def score(self, puzzle: Puzzle, guess: str) -> Tuple[float, str]:
        sys = (
            "你是海龟汤评测裁判。根据真实汤底，评估玩家还原的准确度，"
            "给出 0-100 的整数分和一句简短评语。只输出 JSON："
            '{"score": <int>, "rationale": "<str>"}'
        )
        conv = [
            {"role": "user", "content": f"真实汤底：{puzzle.solution}"},
            {"role": "user", "content": f"玩家还原：{guess}"},
        ]
        try:
            raw = _call(self.model, sys, conv, temperature=0.2, max_tokens=200)
        except LLMError:
            return 0.0, "（裁判无响应，未评分）"
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            obj = json.loads(raw[start:end])
            return float(obj.get("score", 0)), str(obj.get("rationale", ""))
        except Exception:
            return 0.0, f"（裁判返回无法解析：{raw[:120]}）"


# ---------- 工厂 ----------
def make_player(spec: str, role: str = "questioner", index: int = 0) -> Player:
    """根据模型名 spec 构造玩家。mock- 开头走 Mock，否则走 LLM。"""
    if spec.startswith("mock-"):
        from .mock import MockQuestioner, MockHost, MockJudge
        if role == "host":
            return MockHost()
        if role == "judge":
            return MockJudge()
        return MockQuestioner(id=spec, model=spec, seed=index)
    if role == "host":
        return LLMHost(model=spec)
    if role == "judge":
        return LLMJudge(model=spec)
    return LLMQuestioner(id=spec, model=spec)
