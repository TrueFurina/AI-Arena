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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, render_template_string, send_file, jsonify

from engine.game import TurtleSoupGame, Puzzle
from engine.puzzles import load_puzzles, get_puzzle
from players import make_player
from analysis.report import build_report

app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
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
<div class="hint">模型名格式 <code>provider/model</code>（如 <code>zai/glm-4.7</code>）；以 <code>mock-</code> 开头为离线内置模型。<code>ashbwin</code> 为免费端点。</div>

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
    return send_file(os.path.join(LOGS, name))


@app.route("/logs/<path:name>")
def log_file(name):
    return send_file(os.path.join(LOGS, name), mimetype="application/json")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8860))
    app.run(host="0.0.0.0", port=port, debug=False)
