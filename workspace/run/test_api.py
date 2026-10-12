"""API 连通性自检 (API health check)。

读取 .env（python-dotenv），对 .env 中配了 key 的 LLM 供应商各发一个最小 chat
completion，报告：是否已配置 / 成功失败 / 延迟(ms) / 模型首句回复。
零 key 也能跑：只报告「未配置」，不抛错、退出码 0。

可选：若设了 TOKENC_API_KEY，额外探测 tokenc.top 网关
（翻译服务作为存活性证明；chat/llm 预期不可用，给出明确结论）。

用法：
  python run/test_api.py                  # 测全部已配置供应商 + tokenc（若设 key）
  python run/test_api.py --model deepseek # 只测某个别名或完整 litellm 名
  python run/test_api.py --skip-tokenc     # 不测 tokenc
  python run/test_api.py --timeout 15
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from players.llm_player import MODEL_ALIASES

# 别名 -> 需要的 env key（用于「未配置」前置判断，避免无谓请求）
ALIAS_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}
# litellm 完整模型名前缀 -> env key（如 openai/gpt-4o -> OPENAI_API_KEY）
PREFIX_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
}

PROBE_MSG = "用一句话中文回复：连通性测试 OK"


def resolve(model):
    """返回 (resolved_litellm_name, needed_env_var)。"""
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model], ALIAS_ENV.get(model)
    needed = PREFIX_ENV.get(model.split("/")[0]) if "/" in model else None
    return model, needed


def test_llm(model, timeout):
    import litellm

    resolved, needed = resolve(model)
    if needed and not os.getenv(needed):
        return {"model": model, "resolved": resolved, "status": "SKIP",
                "reason": f"未配置 {needed}", "ms": 0, "reply": ""}
    try:
        t0 = time.monotonic()
        resp = litellm.completion(
            model=resolved,
            messages=[{"role": "user", "content": PROBE_MSG}],
            max_tokens=30,
            temperature=0,
            timeout=timeout,
        )
        ms = int((time.monotonic() - t0) * 1000)
        reply = resp.choices[0].message.content.strip().replace("\n", " ")
        return {"model": model, "resolved": resolved, "status": "OK",
                "reason": "", "ms": ms, "reply": reply[:60]}
    except litellm.AuthenticationError:
        return {"model": model, "resolved": resolved, "status": "FAIL",
                "reason": "鉴权失败（key 无效/缺失）", "ms": 0, "reply": ""}
    except litellm.BadRequestError as e:
        return {"model": model, "resolved": resolved, "status": "FAIL",
                "reason": f"请求参数错误: {str(e)[:80]}", "ms": 0, "reply": ""}
    except litellm.RateLimitError:
        return {"model": model, "resolved": resolved, "status": "FAIL",
                "reason": "限流 (RateLimit)", "ms": 0, "reply": ""}
    except litellm.Timeout:
        return {"model": model, "resolved": resolved, "status": "FAIL",
                "reason": f"超时 (>={timeout}s)", "ms": 0, "reply": ""}
    except Exception as e:  # 网络/其他
        return {"model": model, "resolved": resolved, "status": "FAIL",
                "reason": f"{type(e).__name__}: {str(e)[:80]}", "ms": 0, "reply": ""}


def _http_post(url, token, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def probe_tokenc(token, timeout):
    """探测 tokenc.top 网关：翻译(存活性) + chat(预期不可用)。"""
    base = "https://tokenc.top/v1"
    out = []
    # 1) 翻译服务作为存活性证明
    try:
        st, body = _http_post(f"{base}/translate", token,
                              {"text": "hello", "to": "zh"}, timeout)
        ok = ("success" in body and "true" in body)
        out.append({
            "service": "translate", "status": "OK" if ok else "WARN",
            "reason": f"HTTP {st} -> {'alive' if ok else body[:80]}",
        })
    except urllib.error.HTTPError as e:
        out.append({"service": "translate", "status": "FAIL",
                    "reason": f"HTTP {e.code} {e.read().decode('utf-8','replace')[:80]}"})
    except Exception as e:
        out.append({"service": "translate", "status": "FAIL",
                    "reason": f"{type(e).__name__}: {str(e)[:80]}"})
    # 2) chat 服务（预期 PROVIDER_NOT_FOUND）
    try:
        st, body = _http_post(f"{base}/chat/completions", token,
                              {"model": "gpt-4o-mini",
                               "messages": [{"role": "user", "content": "hi"}]},
                              timeout)
        out.append({
            "service": "chat(LLM)", "status": "UNEXPECTED",
            "reason": f"HTTP {st} -> {body[:80]}",
        })
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        out.append({
            "service": "chat(LLM)", "status": "EXPECTED_OFF",
            "reason": f"HTTP {e.code} -> {body[:80]}",
        })
    except Exception as e:
        out.append({"service": "chat(LLM)", "status": "FAIL",
                    "reason": f"{type(e).__name__}: {str(e)[:80]}"})
    return out


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI-Arena API 连通性自检")
    parser.add_argument("--model", default=None,
                        help="只测某个别名或完整 litellm 名（如 deepseek / openai/gpt-4o）")
    parser.add_argument("--skip-tokenc", action="store_true", help="不探测 tokenc 网关")
    parser.add_argument("--timeout", type=int, default=20, help="单请求超时秒数")
    args = parser.parse_args()

    print("=" * 64)
    print("AI-Arena · API 连通性自检")
    print("=" * 64)

    models = [args.model] if args.model else list(MODEL_ALIASES.keys()) + [
        "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"]
    print("\n--- LLM 供应商 (litellm) ---")
    any_ok = False
    for m in models:
        r = test_llm(m, args.timeout)
        flag = {"OK": "✅", "FAIL": "❌", "SKIP": "⚪"}.get(r["status"], "·")
        line = f"{flag} {r['model']:<28} [{r['status']}]"
        if r["status"] == "OK":
            any_ok = True
            line += f"  {r['ms']}ms  «{r['reply']}»"
        elif r["status"] != "SKIP":
            line += f"  {r['reason']}"
        else:
            line += f"  ({r['reason']})"
        print(line)

    if not args.skip_tokenc and os.getenv("TOKENC_API_KEY"):
        print("\n--- tokenc.top 网关（通用工具 API，非 LLM）---")
        for r in probe_tokenc(os.getenv("TOKENC_API_KEY"), args.timeout):
            print(f"· {r['service']:<14} [{r['status']}]  {r['reason']}")
        print("  结论：tokenc.top 是通用工具网关，不提供 LLM 对话，不能作为本项目的模型后端。")
    elif not args.skip_tokenc:
        print("\n--- tokenc.top 网关 ---")
        print("· 未配置 TOKENC_API_KEY，跳过（可在 .env 设入以联调探测）。")

    print("\n" + "=" * 64)
    if any_ok:
        print("结果：已有可用 LLM key，可直接跑  python run/single_game.py --mode llm --model <别名>")
    else:
        print("结果：暂无可用的 LLM key（全部 SKIP/FAIL）。填入真实 key 到 .env 后即可跑真实对局。")
    print("=" * 64)


if __name__ == "__main__":
    main()
