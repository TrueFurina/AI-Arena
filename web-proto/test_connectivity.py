#!/usr/bin/env python3
"""
AI Arena · 多供应商网络连通性自检
==================================

用途：在「平台所在沙箱（cloudstudio）」上跑一条命令，立刻知道
      哪些 LLM 供应商从当前网络可达，避免把时间浪费在不可达的端点上。

特点：
  - 纯网络探测（TCP 连接 + HTTPS 握手），不依赖具体 model 名、不发送任何密钥。
  - 同时覆盖项目里接线过的所有供应商：ashbwin / zai / moonshot / deepseek /
    openai / qwen，以及自定义 ASHBWIN_API_BASE。
  - 自动读取 .env，仅用于提示「哪些供应商已配 key」（key 内容绝不打印）。

用法：
    python test_connectivity.py

依赖：pip install requests python-dotenv
"""
import os
import socket
import ssl
import sys

try:
    import requests
except ImportError:
    print("缺少依赖：pip install requests")
    sys.exit(2)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 供应商 -> (主机, 端口, scheme)。均为各家公开 API 主机。
DEFAULT_HOSTS = {
    "ashbwin":  ("203.135.99.69", 11000, "http"),
    "zai":      ("open.bigmodel.cn", 443, "https"),
    "moonshot": ("api.moonshot.cn", 443, "https"),
    "deepseek": ("api.deepseek.com", 443, "https"),
    "openai":   ("api.openai.com", 443, "https"),
    "qwen":     ("dashscope.aliyuncs.com", 443, "https"),
}

# 若 .env 里自定义了 ashbwin base，解析主机端口覆盖默认
ash_base = os.getenv("ASHBWIN_API_BASE", "")
if ash_base:
    try:
        from urllib.parse import urlparse
        p = urlparse(ash_base)
        host = p.hostname
        port = p.port or (443 if p.scheme == "https" else 80)
        if host:
            DEFAULT_HOSTS["ashbwin"] = (host, port, p.scheme or "http")
    except Exception:
        pass

# 哪些供应商在 .env 里配了 key（只记“有没有”，不记值）
KEY_ENV = {
    "ashbwin": "ASHBWIN_API_KEY",
    "zai": "ZAI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
}


def tcp_probe(host: str, port: int, timeout: float = 6.0):
    """返回 (ok: bool, msg: str)。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "TCP 可连"
    except socket.timeout:
        return False, "连接超时"
    except ConnectionRefusedError:
        return False, "连接被拒绝 (Connection refused)"
    except OSError as e:
        return False, f"网络错误: {e}"


def https_probe(host: str, port: int, timeout: float = 8.0):
    """TCP 之上做 TLS 握手 + 一次最小 GET，区分『通但需鉴权』与『纯不通』。"""
    ok, msg = tcp_probe(host, port, timeout)
    if not ok:
        return ok, msg
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout):
            with ctx.wrap_socket(socket.socket(), server_hostname=host) as s:
                s.connect((host, port))
                # 已建立 TLS，说明主机可达且证书有效
                return True, "TLS 握手成功（主机可达）"
    except ssl.SSLError as e:
        return True, f"TCP 通但 TLS 异常: {e}"
    except socket.timeout:
        return False, "TLS 握手超时"
    except OSError as e:
        return False, f"TLS 阶段网络错误: {e}"


def http_get_probe(host: str, port: int, scheme: str, timeout: float = 8.0):
    """用 requests 发最小 GET，看返回状态（401/403 也算『可达』）。"""
    url = f"{scheme}://{host}:{port}/"
    try:
        r = requests.get(url, timeout=timeout, stream=True, verify=True)
        return True, f"HTTP 可达 (状态码 {r.status_code})"
    except requests.exceptions.SSLError as e:
        return True, f"TCP 通但 SSL 错误: {e}"
    except requests.exceptions.ConnectionError as e:
        return False, f"连接失败: {e}"
    except requests.exceptions.Timeout:
        return False, "请求超时"
    except Exception as e:
        return False, f"异常: {e}"


def probe_one(name: str, host: str, port: int, scheme: str):
    if scheme == "https":
        ok, msg = https_probe(host, port)
        if ok and "TLS 握手成功" in msg:
            # 再补一次 HTTP 层确认
            _ok, _msg = http_get_probe(host, port, scheme)
            if _ok:
                return True, _msg
            return True, msg + "（HTTP 层未进一步确认，但主机可达）"
        return ok, msg
    else:
        ok, msg = tcp_probe(host, port)
        if ok:
            _ok, _msg = http_get_probe(host, port, scheme)
            if _ok:
                return True, _msg
        return ok, msg


def main():
    print("=== AI Arena · 供应商网络连通性自检 ===\n")
    has_key = {n: (os.getenv(KEY_ENV[n], "") != "") for n in KEY_ENV}
    rows = []
    for name, (host, port, scheme) in DEFAULT_HOSTS.items():
        ok, msg = probe_one(name, host, port, scheme)
        key_mark = "✓已配key" if has_key.get(name) else "—无key"
        rows.append((name, f"{host}:{port}", scheme, "可达✅" if ok else "不可达❌", msg, key_mark))

    # 打印表格
    w = [10, 34, 6, 9, 40, 10]
    hdr = f"{'供应商':<{w[0]}} {'主机:端口':<{w[1]}} {'协议':<{w[2]}} {'结果':<{w[3]}} {'说明':<{w[4]}} {'key':<{w[5]}}"
    print(hdr)
    print("-" * (sum(w) + 6))
    for r in rows:
        print(f"{r[0]:<{w[0]}} {r[1]:<{w[1]}} {r[2]:<{w[2]}} {r[3]:<{w[3]}} {r[4]:<{w[4]}} {r[5]:<{w[5]}}")

    reachable = [r[0] for r in rows if r[3].startswith("可达")]
    print("\n=== 结论 ===")
    if reachable:
        print(f"✅ 当前网络可达：{', '.join(reachable)}")
        print("   这些供应商可在平台上尝试接入（仍需有效 key 与正确 model 名）。")
    else:
        print("❌ 当前网络下所有供应商均不可达。")
    if not any(r[3].startswith("可达") for r in rows):
        print("   建议：在平台（cloudstudio）沙箱重跑本脚本；若那里可达而本沙箱不可达，")
        print("   说明端点做了来源白名单，这是预期行为。")


if __name__ == "__main__":
    main()
