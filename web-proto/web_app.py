#!/usr/bin/env python3
"""
AI Arena 海龟汤评测平台（Web 界面）。

对应 API使用教程.md 描述的 UI：
- 出题人填汤面/汤底，或选内置汤底
- 提问者（AI）填模型名，可添加多个（mock- 开头为内置离线模型）
- API Key 留空自动读 .env
- 点「开始评测」输出各 AI 还原对比报告 + 评分图

运行：python web_app.py  ->  http://localhost:8860
（生产由 supervisord 托管，端口 8860）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, render_template_string, send_from_directory, jsonify

from engine.game import TurtleSoupGame, Puzzle
from engine.puzzles import load_puzzles, get_puzzle
from players import make_player
from analysis.report import build_report

# 凭证层软导入：验签是**附加**能力，导入失败不得拖垮既有评测页。
# 但 /verify 会显式报「不可用」——静默降级成假页面比报错危险（项目既有教训）。
try:
    from chain.verify import verify_credential
    from chain.models import Credential
    from chain.identity import IdentityRegistry
    CHAIN_OK = True
    CHAIN_ERR = ""
except Exception as _exc:  # 不吞信息：错误原样透给 /verify 页面
    CHAIN_OK = False
    CHAIN_ERR = f"{type(_exc).__name__}: {_exc}"

app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
CRED_DIR = os.path.join(LOGS, "credentials")
os.makedirs(LOGS, exist_ok=True)

INDEX_HTML = """
<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>AI Arena · 海龟汤评测</title>
<style>
  body{font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;max-width:860px;margin:24px auto;padding:0 16px;background:#f7f8fa;color:#222}
  h1{font-size:22px} .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 18px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
  label{font-weight:600;display:block;margin:10px 0 4px}
  input,textarea,select{width:100%;padding:8px 10px;border:1px solid #d1d5db;border-radius:8px;box-sizing:border-box;font-size:14px}
  textarea{height:70px;resize:vertical}
  .row{display:flex;gap:8px;align-items:center;margin:6px 0}
  .row input{flex:1}
  button{cursor:pointer;border:none;border-radius:8px;padding:9px 14px;font-size:14px;font-weight:600}
  .primary{background:#4C72B0;color:#fff}
  .ghost{background:#eef2f7;color:#334}
  .danger{background:#fde8e8;color:#b91c1c}
  .qrow{display:flex;gap:8px;margin:6px 0}
  .qrow input{flex:1}
  .hint{color:#6b7280;font-size:12px}
  pre{white-space:pre-wrap;background:#f3f4f6;padding:12px;border-radius:8px;font-size:13px}
  img{max-width:100%;border:1px solid #e5e7eb;border-radius:8px;margin-top:8px}
  .score{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0}
  .score div{background:#eef2f7;border-radius:8px;padding:8px 12px}
</style></head>
<body>
<h1>🐢 AI Arena · 海龟汤 AI 对战评测</h1>
<div class="hint">模型名格式 <code>provider/model</code>（如 <code>zai/glm-4.7</code>）；以 <code>mock-</code> 开头为离线内置模型。<code>ashbwin</code> 为免费端点。<br>
<a href="/verify">🔏 凭证验签入口（粘贴凭证或按 tx_id 检索）</a></div>

<form id="f" method="post" action="/api/evaluate">
  <div class="card">
    <label>选择内置汤底（或下面自定义）</label>
    <select name="puzzle_id">
      <option value="">— 自定义 —</option>
      {% for p in puzzles %}<option value="{{p.soup_id}}">{{p.title}} ({{p.soup_id}})</option>{% endfor %}
    </select>
    <label>汤面（谜面）</label>
    <textarea name="surface" placeholder="一个人走进餐厅，点了一碗海龟汤……"></textarea>
    <label>汤底（真相，留空则用内置）</label>
    <textarea name="solution" placeholder="完整真相……"></textarea>
  </div>

  <div class="card">
    <label>提问者（AI 玩家）</label>
    <div id="qs">
      <div class="qrow"><input name="questioners" placeholder="如 zai/glm-4.5-flash 或 mock-gpt4o"></div>
    </div>
    <button type="button" class="ghost" onclick="addQ()">＋ 添加提问者</button>
    <div class="row"><div>
      <label>主持（汤主）模型</label>
      <input name="host" value="mock-host" placeholder="mock-host / zai/glm-4.7">
    </div><div>
      <label>裁判模型</label>
      <input name="judge" value="mock" placeholder="mock / openai/gpt-4o-mini">
    </div><div>
      <label>轮数</label>
      <input name="rounds" value="8" type="number">
    </div></div>
    <button type="submit" class="primary" style="margin-top:12px">⚡ 开始评测</button>
  </div>
</form>

<script>
function addQ(){const d=document.createElement('div');d.className='qrow';d.innerHTML='<input name=\"questioners\" placeholder=\"模型名\"><button type=\"button\" class=\"danger\" onclick=\"this.parentNode.remove()\">✕</button>';document.getElementById('qs').appendChild(d);}
</script>
</body></html>
"""

RESULT_HTML = """
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>评测结果 · AI Arena</title>
<style>body{font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;max-width:860px;margin:24px auto;padding:0 16px;background:#f7f8fa;color:#222}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 18px;margin:14px 0}
.score{display:flex;gap:18px;flex-wrap:wrap;margin:8px 0}.score div{background:#eef2f7;border-radius:8px;padding:8px 12px}
pre{white-space:pre-wrap;background:#f3f4f6;padding:12px;border-radius:8px;font-size:13px}
img{max-width:100%;border:1px solid #e5e7eb;border-radius:8px}
a{color:#4C72B0}</style></head><body>
<h1>🐢 评测结果</h1>
<div class="card"><div class="score">{% for s in scores %}<div><b>{{s.player}}</b><br>{{s.score}} 分<br><span style="font-size:12px;color:#6b7280">{{s.rationale}}</span></div>{% endfor %}</div>
<p>胜者：<b>{{winner or '（无）'}}</b> ｜ <a href="/report/report_{{game_id}}.png">评分图</a> ｜ <a href="/logs/game_{{game_id}}.json">原始日志</a></p>
<img src="/report/report_{{game_id}}.png"></div>
<div class="card"><h3>对局记录</h3><pre>{{transcript_text}}</pre></div>
<div class="card"><h3>汤底（真相）</h3><pre>{{solution}}</pre></div>
<a href="/">← 返回</a></body></html>
"""


VERIFY_HTML = """
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>凭证验签 · AI Arena</title>
<style>body{font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;max-width:900px;margin:24px auto;padding:0 16px;background:#f7f8fa;color:#222}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 18px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
h1{font-size:22px}h3{margin:0 0 8px}
textarea,input{width:100%;padding:8px 10px;border:1px solid #d1d5db;border-radius:8px;box-sizing:border-box;font-size:13px}
textarea{height:150px;font-family:ui-monospace,Consolas,monospace;resize:vertical}
button{cursor:pointer;border:none;border-radius:8px;padding:9px 14px;font-size:14px;font-weight:600;background:#4C72B0;color:#fff}
.ghost{background:#eef2f7;color:#334}
.hint{color:#6b7280;font-size:12px;line-height:1.6}
pre{white-space:pre-wrap;background:#f3f4f6;padding:12px;border-radius:8px;font-size:12px}
.ok{color:#15803d;font-weight:700}.bad{color:#b91c1c;font-weight:700}.warn{color:#b45309;font-weight:700}
.row{display:flex;gap:10px;align-items:center;margin:8px 0}
a{color:#4C72B0}</style></head><body>
<h1>🔏 评测凭证验签</h1>
<div class="card"><div class="hint">
凭证是<b>自包含</b>的（含记录 + 公钥 + 签名），验签在你提交的这一份内容上完成，<b>不需要访问本项目的运行环境或密钥库</b>。<br>
⚠️ 边界：<b>tx_id 只是链上交易索引，凭一个 tx_id 号无法验任何东西</b>。tx_id 入口只在本机凭证库
<code>logs/credentials/</code> 中检索出对应凭证，再对那份凭证做同样完整的验签；库中没有就如实报「未命中」，<b>不会</b>显示为验签失败。
</div></div>

{% if not chain_ok %}
<div class="card"><b class="bad">凭证层不可用</b><pre>{{chain_err}}</pre>
<div class="hint">既有评测页不受影响；此处不做静默降级。</div></div>
{% else %}

<div class="card">
<h3>方式一 · 粘贴凭证 JSON（完整离线验签）</h3>
<textarea id="cred" placeholder='把 *.credential.json 的全部内容粘贴到这里'></textarea>
<div class="row"><label><input type="checkbox" id="ident1" style="width:auto"> 同时做身份锚定校验（比对签发者公钥与身份合约登记）</label></div>
<button onclick="run({mode:'credential'})">验签</button>
</div>

<div class="card">
<h3>方式二 · 按 tx_id 检索（本机凭证库）</h3>
<input id="txid" placeholder="如 e7c217f4321afa51dcbe2ba62a04c458">
<div class="row"><label><input type="checkbox" id="ident2" style="width:auto"> 同时做身份锚定校验</label></div>
<button class="ghost" onclick="run({mode:'tx_id'})">检索并验签</button>
</div>
{% endif %}

<div class="card" id="out" style="display:none"><h3>结果</h3><div id="res"></div></div>
<p><a href="/">← 返回评测页</a></p>

<script>
function run(cfg){
  const body = cfg.mode==='credential'
    ? {credential: document.getElementById('cred').value, identity: document.getElementById('ident1').checked}
    : {tx_id: document.getElementById('txid').value.trim(), identity: document.getElementById('ident2').checked};
  fetch('/api/verify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(r=>r.json()).then(show).catch(e=>show({error:String(e)}));
}
function show(d){
  const o=document.getElementById('out'); o.style.display='block';
  if(d.error){document.getElementById('res').innerHTML='<b class="bad">请求失败</b><pre>'+d.error+'</pre>';return;}
  let h='';
  if(d.mode==='tx_id' && d.found===false){
    h+='<p class="warn">⚠️ 本机凭证库中未找到该 tx_id —— 这是「未命中」，<b>不是</b>验签失败。</p>';
    h+='<div class="hint">'+d.reason+'</div>';
    document.getElementById('res').innerHTML=h; return;
  }
  if(d.ok===true) h+='<p class="ok">✅ 凭证可信</p>';
  else if(d.ok===false) h+='<p class="bad">❌ 凭证不可信（记录被篡改或签名不匹配）</p>';
  else h+='<p class="warn">⚠️ 未执行验签</p>';
  if(d.summary) h+='<pre>'+d.summary+'</pre>';
  if(d.steps){h+='<h3>逐项</h3><ul>';
    for(const k in d.steps){h+='<li>'+k+'：'+(d.steps[k]?'<span class="ok">通过</span>':'<span class="bad">失败</span>')+'</li>';}
    h+='</ul>';}
  if(d.identity_status) h+='<p>身份锚定：<b>'+d.identity_status+'</b></p>';
  else h+='<p class="hint">身份锚定：未执行（未勾选或合约不可用）—— 未接身份锚定时，任何人都能自造密钥签发自洽的假凭证且验签通过。</p>';
  if(d.reasons && Object.keys(d.reasons).length){h+='<h3>原因</h3><ul>';
    for(const k in d.reasons){h+='<li>'+d.reasons[k]+'</li>';} h+='</ul>';}
  document.getElementById('res').innerHTML=h;
}
</script>
</body></html>
"""


def _credential_index() -> dict:
    """扫本机凭证库，建 tx_id / match_id / record_hash → 路径 的索引。

    只在**本地已签发**的凭证里检索：本服务没有、也不应有能力凭 tx_id
    去链上取回一份没见过的凭证再做校验。
    """
    idx: dict = {}
    if not os.path.isdir(CRED_DIR):
        return idx
    for name in sorted(os.listdir(CRED_DIR)):
        if not name.endswith(".credential.json"):
            continue
        p = os.path.join(CRED_DIR, name)
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        anchor = d.get("anchor") or {}
        rec = d.get("record") or {}
        for key in (anchor.get("tx_id"), rec.get("match_id"), d.get("record_hash")):
            if key:
                idx.setdefault(str(key), p)
    return idx


@app.route("/")
def index():
    return render_template_string(INDEX_HTML, puzzles=load_puzzles())


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    form = request.form
    puzzle_id = form.get("puzzle_id", "").strip()
    surface = form.get("surface", "").strip()
    solution = form.get("solution", "").strip()
    questioners = [x.strip() for x in form.getlist("questioners") if x.strip()]
    host_spec = form.get("host", "mock-host").strip() or "mock-host"
    judge_spec = form.get("judge", "mock").strip() or "mock"
    rounds = int(form.get("rounds", 8) or 8)

    if not questioners:
        questioners = ["mock-gpt4o"]

    # 题库或自定义
    if puzzle_id:
        puzzle = get_puzzle(puzzle_id)
    else:
        puzzle = Puzzle(
            soup_id="custom", title="自定义题目", surface=surface or "（未填写汤面）",
            solution=solution or "（未填写汤底）", golden_qa=[], tags=[],
        )

    qs = [make_player(spec, role="questioner", index=i) for i, spec in enumerate(questioners)]
    host = make_player(host_spec, role="host")
    judge = make_player(judge_spec, role="judge") if judge_spec != "mock" else make_player("mock-judge", role="judge")

    game = TurtleSoupGame(puzzle, host, qs, judge=judge, max_rounds=rounds)
    result = game.run()
    result.save(os.path.join(LOGS, f"game_{result.game_id}.json"))
    build_report(result, LOGS)

    transcript_text = "\n".join(
        (f"[{m['round']}] {m['speaker']}({m['role']}): {m['content']}") for m in result.transcript
    )
    return render_template_string(
        RESULT_HTML, scores=result.scores, winner=result.winner,
        game_id=result.game_id, transcript_text=transcript_text,
        solution=result.puzzle.get("solution", ""),
    )


@app.route("/report/<path:name>")
def report_file(name):
    # 安全（2026-10-13）：曾用 send_file(os.path.join(LOGS, name))——name 含
    # "../" 可穿越到 .env / data/chain_keys/*.pem。send_from_directory 内部
    # safe_join 拒绝穿越（越界即 404），是 Flask 官方的安全形态
    return send_from_directory(LOGS, name)


@app.route("/logs/<path:name>")
def log_file(name):
    return send_from_directory(LOGS, name, mimetype="application/json")


@app.route("/verify")
def verify_page():
    return render_template_string(VERIFY_HTML, chain_ok=CHAIN_OK,
                                  chain_err=CHAIN_ERR, cred_dir=CRED_DIR)


@app.route("/api/verify", methods=["POST"])
def api_verify():
    """验签入口。两种输入，语义不同且不得混淆：

      credential → 完整离线验签（ok=true/false）
      tx_id      → 本机凭证库检索；未命中返回 found=false + ok=null
                   （**未做验签**，绝不能显示成「验签失败」）
    """
    if not CHAIN_OK:
        return jsonify({"error": f"凭证层不可用：{CHAIN_ERR}"}), 503

    data = request.get_json(silent=True)
    if not data:
        data = request.form.to_dict() if request.form else {}
    cred_text = str(data.get("credential") or "").strip()
    tx_id = str(data.get("tx_id") or "").strip()
    use_identity = str(data.get("identity", "")).lower() in ("1", "true", "on", "yes")

    if not cred_text and not tx_id:
        return jsonify({"error": "请提供凭证 JSON 或 tx_id（二者至少其一）"}), 400

    if cred_text:
        mode, found = "credential", True
        try:
            cred = Credential.from_json(cred_text)
        except Exception as exc:
            return jsonify({"mode": "credential", "found": True, "ok": False,
                            "reason": f"凭证解析失败：{type(exc).__name__}: {exc}"}), 400
    else:
        mode = "tx_id"
        hit = _credential_index().get(tx_id)
        if hit is None:
            return jsonify({
                "mode": "tx_id", "found": False, "ok": None,
                "reason": (f"本机凭证库（{CRED_DIR}）中没有 tx_id={tx_id} 对应的凭证。"
                           "tx_id 只是链上索引，凭它无法凭空验签——"
                           "这是「未命中」，不是「验签失败」。"),
            })
        try:
            cred = Credential.from_dict(
                json.loads(Path(hit).read_text(encoding="utf-8")))
        except Exception as exc:
            return jsonify({"mode": "tx_id", "found": True, "ok": False,
                            "reason": f"凭证读取失败：{type(exc).__name__}: {exc}"}), 500
        found = True

    identity = IdentityRegistry() if use_identity else None
    vr = verify_credential(cred, identity=identity)
    rec = cred.record or {}
    return jsonify({
        "mode": mode, "found": found, "ok": vr.ok,
        "steps": vr.steps, "reasons": vr.reasons,
        "identity_status": vr.identity_status,
        "anchor_status": vr.anchor_status, "key_custody": vr.key_custody,
        "summary": vr.summary(),
        "match_id": rec.get("match_id"), "puzzle_id": rec.get("puzzle_id"),
        "record_hash": cred.record_hash,
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8860))
    app.run(host="0.0.0.0", port=port, debug=False)
