#!/usr/bin/env python3
"""
AI Arena · 真实模型对话实测（最小调用）
======================================
用 .env 里已有的 zai(Zhipu BigModel) / moonshot key，
直接打各家的 OpenAI 兼容 /chat/completions 端点，
验证「平台能否跑真实模型」——与 ashbwin 无关。

不依赖 litellm，只用 requests（系统已带）。
每个供应商只发一次极小调用（max_tokens=30），成本可忽略。

用法：python3 test_real_models.py
"""
import os
import json

try:
    import requests
except ImportError:
    print("缺少 requests"); raise SystemExit(2)


def load_env(path=".env"):
    """手动解析 .env（不打印任何值）。"""
    env = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def try_chat(name, url, api_key, model, timeout=30):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "用四个字以内回答：1+1=?"}],
        "max_tokens": 30,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code == 200:
            js = r.json()
            content = js["choices"][0]["message"]["content"]
            return True, f"✅ {model} 回复: {content!r}"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"异常: {e!r}"


def main():
    env = load_env()
    print("=== 真实模型对话实测 ===\n")

    # 供应商候选：(展示名, 端点, key变量, [候选model...])
    candidates = [
        ("moonshot", "https://api.moonshot.cn/v1/chat/completions", "MOONSHOT_API_KEY",
         ["moonshot-v1-8k", "moonshot-v1-32k", "kimi-k2"]),
        ("zai(Zhipu BigModel)", "https://open.bigmodel.cn/api/paas/v4/chat/completions", "ZAI_API_KEY",
         ["glm-4-flash", "glm-4-plus", "glm-4.5-air"]),
    ]

    any_ok = False
    for disp, url, keyvar, models in candidates:
        key = env.get(keyvar, "")
        if not key:
            print(f"[{disp}] 未找到 {keyvar}，跳过")
            continue
        print(f"[{disp}] 端点 {url}")
        ok_any = False
        for m in models:
            ok, msg = try_chat(disp, url, key, m)
            print(f"   · model={m:18s} -> {msg}")
            if ok:
                ok_any = True
                any_ok = True
                break
        if not ok_any:
            print(f"   ⚠️ 该供应商尝试的 model 名均未成功（可能 model 名需按平台文档调整）")
        print()

    print("=== 结论 ===")
    if any_ok:
        print("✅ 至少一家真实模型可用！平台无需依赖 ashbwin 即可跑真实对战。")
        print("   在 web_app.py 的提问者/裁判里填对应 provider/model（如 moonshot/moonshot-v1-8k）即可。")
    else:
        print("⚠️ 本轮真实模型调用未成功。可能原因：model 名与平台文档不符 / key 失效 / 需特定 base_url。")
        print("   请对照 API_KEYS.md 校正 model 名与端点；或确认 key 仍有效。")


if __name__ == "__main__":
    main()
