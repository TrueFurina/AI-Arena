#!/usr/bin/env python3
"""
ashbwin 免费端点 · 通断 + 能力自检脚本
========================================
用途：在 AI-Arena 平台所在的沙箱（cloudstudio）上跑，验证
      http://203.135.99.69:11000/v1 这个免费 API 到底通不通、
      ashbwin 模型能不能正常对话。

依赖：pip install requests

用法：python test_ashbwin.py
（端点地址 / key 也可通过环境变量覆盖：
 ASHBWIN_BASE、ASHBWIN_KEY、ASHBWIN_MODEL）
"""
import os
import sys
import json
import requests

BASE = os.getenv("ASHBWIN_BASE", "http://203.135.99.69:11000/v1")
KEY = os.getenv("ASHBWIN_KEY", "sk-8db30db0-6e4d-4749-8fe7-96cc261dd7aa")
MODEL = os.getenv("ASHBWIN_MODEL", "ashbwin")

H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

def line(t=""):
    print(t)

def test_models():
    line("\n=== 1) GET /v1/models ===")
    try:
        r = requests.get(f"{BASE}/models", headers=H, timeout=20)
        line(f"HTTP {r.status_code}")
        try:
            js = r.json()
            ids = [m.get("id") for m in js.get("data", [])]
            line("可用模型: " + (", ".join(ids) if ids else str(js)[:300]))
        except Exception:
            line(r.text[:400])
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        line("✗ 连接被对端静默关闭（空响应）—— 典型：网络白名单 / 后端未起 / WAF 拦截机房IP")
        return False
    except Exception as e:
        line(f"✗ 异常: {e!r}")
        return False

def test_chat():
    line("\n=== 2) POST /v1/chat/completions ===")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一个推理助手，请用一句话回答。"},
            {"role": "user", "content": "海龟汤里，一个人走进餐厅点了一碗海龟汤，喝了一口后自杀了。为什么？请给一个简短的推理方向。"},
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    try:
        r = requests.post(f"{BASE}/chat/completions", headers=H, json=payload, timeout=60)
        line(f"HTTP {r.status_code}")
        if r.status_code == 200:
            js = r.json()
            content = js["choices"][0]["message"]["content"]
            line("✓ 模型回复:\n" + content)
            return True
        else:
            line("✗ 响应体: " + r.text[:500])
            return False
    except requests.exceptions.ConnectionError:
        line("✗ 连接被静默关闭（空响应）—— 端点对当前来源不可达")
        return False
    except Exception as e:
        line(f"✗ 异常: {e!r}")
        return False

if __name__ == "__main__":
    line(f"目标端点: {BASE}")
    line(f"模型名:   {MODEL}")
    ok1 = test_models()
    ok2 = test_chat()
    line("\n========== 结论 ==========")
    if ok1 and ok2:
        line("✅ 端点完全可用，可直接接入 AI Arena。")
        sys.exit(0)
    elif ok2:
        line("⚠️ /models 不可用但 chat 可用，多数 OpenAI 兼容端点如此，不影响接入。")
        sys.exit(0)
    else:
        line("❌ 端点当前不可达（空响应）。请确认：")
        line("   1) 该 API 是否只对平台所在沙箱（cloudstudio）放行？")
        line("   2) 端点当前是否在线 / 是否需要特定域名 Host？")
        line("   3) 用浏览器或 curl 从平台沙箱再测一次。")
        sys.exit(1)
