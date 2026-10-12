"""基于 litellm 的真实模型提问者（M1 接入）。对应 specs/engine-design.md §1/§5。

设计要点：
- 通过 litellm 统一接入多家模型（DeepSeek / 通义 Qwen / Kimi / OpenAI / Claude），
  key 由 litellm 按环境变量（DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / ...）自动读取，
  本项目代码**不写死任何 key**（项目规则 2）。
- 输出严格走 JSON 动作协议（specs §5.2）：{"action":"ask"|"guess","content":str}。
- 解析失败自带重试；穷尽重试后返回一个安全的占位 ask，不污染判定（详见 act()）。
- 支持依赖注入 completion_fn，便于无 key 环境下用假模型做冒烟测试。

GM 在 M1 仍由规则化 MockGM 担任（见 mock_player.py）；更聪明的 GM 在 M2 引入。
"""

import json
import os
import re
import sys

from .base import Player

# 友好别名 -> litellm 模型名。--model 可直接传别名。
MODEL_ALIASES = {
    "deepseek": "deepseek/deepseek-chat",
    "qwen": "qwen/qwen-plus",
    "kimi": "moonshot/moonshot-v1-8k",
    "gpt": "openai/gpt-4o-mini",
    "claude": "anthropic/claude-3-5-sonnet-20241022",
}

# 内联兜底提示词（prompts/questioner.md 缺失时使用）。
DEFAULT_PROMPT = """你是海龟汤（情境推理谜题）的提问者。每轮向主持人提一个是非题，用最少轮数还原汤底。
严格只输出 JSON：继续提问 {"action":"ask","content":"中文是非问句"}；自信时 {"action":"guess","content":"完整真相陈述"}。
不要输出 JSON 以外的任何文字。"""


def _default_prompt_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "questioner.md")


class LLMQuestioner(Player):
    """用 litellm 调用真实模型的提问者。"""

    def __init__(self, model, system_prompt=None, max_questions=15,
                 temperature=0.7, max_retries=3, completion_fn=None, **litellm_kwargs):
        self.model = model
        self.name = model  # 用于日志 meta.model
        self.system_prompt = system_prompt or DEFAULT_PROMPT
        self.max_questions = max_questions
        self.temperature = temperature
        self.max_retries = max_retries
        self.kwargs = litellm_kwargs
        # 依赖注入：默认 None → 首次 act 时惰性加载 litellm（见 _ensure_litellm）。
        # 这样零 key 的 mock 路径 / 测试注入假函数时，模块加载不要求装 litellm。
        self.completion_fn = completion_fn
        self._litellm = None
        self._asked = 0
        self.last_error = None

    def _ensure_litellm(self):
        """惰性导入 litellm，仅在真正调用真实模型时触发。

        无 litellm 依赖时，mock 路径 / 测试注入假函数均不受影响；
        只有显式用真实模型且未注入 completion_fn 时才要求安装。
        """
        if self.completion_fn is not None:
            return  # 已注入（真实 litellm 或测试假函数），无需加载
        try:
            import litellm
        except ImportError as e:
            raise RuntimeError(
                "未安装 litellm，无法调用真实模型。请 `pip install litellm`，"
                "或在零 key 模式下使用 --mock / MockQuestioner。"
            ) from e
        self.completion_fn = litellm.completion
        self._litellm = litellm

    @classmethod
    def from_model(cls, model, prompt_path=None, **kwargs):
        """按友好别名或完整 litellm 名解析，并加载提示词。"""
        resolved = MODEL_ALIASES.get(model, model)
        path = prompt_path or _default_prompt_path()
        system_prompt = cls._load_prompt(path)
        return cls(model=resolved, system_prompt=system_prompt, **kwargs)

    @staticmethod
    def _load_prompt(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except (FileNotFoundError, OSError):
            return DEFAULT_PROMPT

    # ---- 提示词构造 ----

    def build_messages(self, observation):
        """把 observation（specs §5.1）拼成 system+user 消息。可单独预览。"""
        transcript = self._format_history(observation.get("history", []))
        user = (
            f"【汤面】{observation.get('surface', '')}\n\n"
            f"【已进行第 {observation.get('turn', 0)} / {observation.get('max_turns', 20)} 轮】\n"
            f"历史问答：\n{transcript if transcript else '（暂无）'}\n\n"
            "请基于以上，输出你这一轮的动作 JSON。"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _format_history(hist):
        """history 是 [{role:questioner,content}, {role:gm,answer}, ...] 交错结构，
        还原成『问/答』逐行文本。"""
        lines = []
        pending_q = None
        for item in hist:
            if item.get("role") == "questioner":
                pending_q = item.get("content", "")
            elif item.get("role") == "gm":
                a = item.get("answer", "")
                lines.append(f"问：{pending_q}\n答：{a}")
                pending_q = None
        return "\n".join(lines)

    # ---- 解析 ----

    @staticmethod
    def _extract_content(resp):
        """兼容 litellm ModelResponse 与裸 dict/str。"""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            return resp["choices"][0]["message"]["content"]
        # litellm ModelResponse
        return resp.choices[0].message.content

    @staticmethod
    def _parse_action(text):
        """从模型文本中抽出第一个 JSON 对象并校验动作协议。失败抛 ValueError。"""
        if not text:
            raise ValueError("空输出")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("未找到 JSON 对象")
        obj = json.loads(m.group())
        action = obj.get("action")
        content = obj.get("content", "")
        if action not in ("ask", "guess"):
            raise ValueError(f"action 非法: {action!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content 为空或非字符串")
        return {"action": action, "content": content.strip()}

    # ---- 主入口 ----

    def act(self, observation):
        if self._asked >= self.max_questions:
            # 达到自设上限，强制猜底收尾，避免死循环。
            return {"action": "guess", "content": "（已达提问上限，主动声明汤底）"}
        self._ensure_litellm()
        llm = self._litellm
        # 仅当真正使用 litellm 时才把鉴权类错误视为致命（立即上抛）；
        # 注入假函数（测试）时不触发此分支。
        fatal = (llm.AuthenticationError, llm.BadRequestError) if llm is not None else ()
        messages = self.build_messages(observation)
        last_err = None
        for _ in range(1, self.max_retries + 1):
            try:
                resp = self.completion_fn(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    **self.kwargs,
                )
                content = self._extract_content(resp)
                action = self._parse_action(content)
                self._asked += 1
                return action
            except fatal:
                # 鉴权失败 / 模型名非法属致命错误，立即上抛，由 single_game 友好提示，不重试。
                raise
            except Exception as e:  # 网络/限流/解析失败等可重试
                last_err = e
        # 穷尽重试：返回安全占位 ask，不污染 guess 判定，由 max_turns 自然收尾。
        self.last_error = last_err
        print(f"[LLMQuestioner] 模型输出解析失败，已重试 {self.max_retries} 次: {last_err}",
              file=sys.stderr)
        self._asked += 1
        return {"action": "ask", "content": "（模型输出无法解析，请忽略此提问）"}

    def preview(self, observation):
        """打印将要发给模型的消息（不调用 API），用于 --dry-run。"""
        print("── 系统提示词 ──")
        print(self.system_prompt)
        print("\n── 用户消息 ──")
        print(self.build_messages(observation)[1]["content"])
