"""web_app 安全守护：/logs/ 与 /report/ 的目录穿越必须被拒。

背景（2026-10-13 部署前评估发现）：两个路由曾用
`send_file(os.path.join(LOGS, name))` 直接拼接用户可控路径——name 含
"../" 可穿越读到 .env（API key）与 data/chain_keys/*.pem（私钥）。
已改 `send_from_directory`（内部 safe_join，越界即 404）。

变异点：把任一路由改回 os.path.join + send_file → 对应测试应变红；
把路由改成「无条件 404」→ 正向对照测试应变红（防假修）。
"""
import os
import sys

import pytest

DECOY_KEY = "ZAI_API_KEY=sk-decoy-not-a-real-key"
DECOY_PEM = "-----BEGIN PRIVATE KEY-----decoy"


def _app():
    pytest.importorskip("flask")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import web_app
    except Exception as exc:  # 依赖缺失时不假装通过
        pytest.skip(f"web_app 不可用：{type(exc).__name__}: {exc}")
    return web_app


def _client():
    return _app().app.test_client()


def _plant_decoys(tmp_path):
    """在 LOGS 之外布置诱饵文件，穿越成功即泄漏。"""
    (tmp_path.parent / ".env").write_text(DECOY_KEY, encoding="utf-8")
    keys = tmp_path.parent / "keys"
    keys.mkdir(exist_ok=True)
    (keys / "secret.pem").write_text(DECOY_PEM, encoding="utf-8")


def _assert_no_leak(resp):
    body = resp.get_data(as_text=True)
    assert DECOY_KEY not in body, ".env 内容被泄漏"
    assert DECOY_PEM not in body, "私钥内容被泄漏"
    assert "root:" not in body, "系统文件内容被泄漏"


# ---------------------------------------------------------------------------
# 正向对照：路由本身要能正常服务 LOGS 内文件（防「一律 404」式假修）
# ---------------------------------------------------------------------------

def test_log_route_still_serves_file_inside_logs(tmp_path, monkeypatch):
    app = _app()
    (tmp_path / "ok.json").write_text('{"hello": 1}', encoding="utf-8")
    monkeypatch.setattr(app, "LOGS", str(tmp_path))
    r = _client().get("/logs/ok.json")
    assert r.status_code == 200
    assert r.get_json() == {"hello": 1}


def test_report_route_still_serves_file_inside_logs(tmp_path, monkeypatch):
    app = _app()
    (tmp_path / "r.md").write_text("# report", encoding="utf-8")
    monkeypatch.setattr(app, "LOGS", str(tmp_path))
    r = _client().get("/report/r.md")
    assert r.status_code == 200
    assert "report" in r.get_data(as_text=True)


# ---------------------------------------------------------------------------
# 负向：各类穿越载荷一律 404 且不泄漏
# ---------------------------------------------------------------------------

ATTACKS = [
    "../.env",                          # 经典相对穿越 → .env（API key）
    "../keys/secret.pem",               # 穿越 → data/chain_keys 类比物（私钥）
    "../../.env",                       # 多级穿越
    "..%2f.env",                        # 编码斜杠变体
    "%2e%2e/keys%2fsecret.pem",         # 全编码点号变体
    "/etc/passwd",                      # 绝对路径
    "....//....//.env",                 # 双写变体
]


@pytest.mark.parametrize("attack", ATTACKS)
def test_log_route_rejects_traversal(tmp_path, monkeypatch, attack):
    app = _app()
    _plant_decoys(tmp_path)
    monkeypatch.setattr(app, "LOGS", str(tmp_path))
    # follow_redirects=True：双斜杠载荷（如 //etc/passwd）会先吃一个 308
    # 合并斜杠重定向，跟随后看终态——必须落在 404 而非 200
    r = _client().get(f"/logs/{attack}", follow_redirects=True)
    assert r.status_code == 404, f"穿越载荷 {attack!r} 未被拒绝（status={r.status_code}）"
    _assert_no_leak(r)


@pytest.mark.parametrize("attack", ["../.env", "..%2fkeys%2fsecret.pem"])
def test_report_route_rejects_traversal(tmp_path, monkeypatch, attack):
    app = _app()
    _plant_decoys(tmp_path)
    monkeypatch.setattr(app, "LOGS", str(tmp_path))
    r = _client().get(f"/report/{attack}", follow_redirects=True)
    assert r.status_code == 404, f"穿越载荷 {attack!r} 未被拒绝（status={r.status_code}）"
    _assert_no_leak(r)
