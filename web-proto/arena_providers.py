#!/usr/bin/env python3
"""
arena_providers.py  ——  AI Arena 自定义模型注册
================================================
把免费端点 ashbwin 注册成一个 litellm 自定义模型，
页面里直接填 "ashbwin" 即可调用，不影响原有的 openai/ 等前缀。

放置位置：AI-Arena 根目录（与 web_app.py 同级）
接入步骤：
  1) 本文件放到 AI-Arena 根目录
  2) 在 web_app.py 顶部加一行：  import arena_providers
  3) .env 里追加下面的 ASHBWIN_* 两行（key 不写死在代码）
  4) 重启服务（supervisord restart）
  5) 提问者/回答者填 "ashbwin" 即可

依赖：pip install litellm python-dotenv
"""
import os
import litellm
from dotenv import load_dotenv

load_dotenv()  # 读取项目根目录 .env

ASHBWIN_BASE = os.getenv("ASHBWIN_API_BASE", "http://203.135.99.69:11000/v1")
ASHBWIN_KEY = os.getenv("ASHBWIN_API_KEY", "sk-8db30db0-6e4d-4749-8fe7-96cc261dd7aa")
ASHBWIN_MODEL = os.getenv("ASHBWIN_MODEL", "ashbwin")


def register_ashbwin():
    """注册自定义 OpenAI 兼容端点为 litellm 模型别名 'ashbwin'。"""
    litellm.register_model({
        ASHBWIN_MODEL: {
            "model_name": ASHBWIN_MODEL,
            "litellm_params": {
                "model": f"openai/{ASHBWIN_MODEL}",
                "api_base": ASHBWIN_BASE,
                "api_key": ASHBWIN_KEY,
            },
        }
    })
    print(f"[arena_providers] 已注册自定义模型 '{ASHBWIN_MODEL}' -> {ASHBWIN_BASE}")


register_ashbwin()
