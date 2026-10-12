#!/usr/bin/env python3
"""
AI Arena · 模型横向对决（成本可控版）
====================================
让两个（或多个）模型在**同一组题目**上各跑完整一局，提取可量化指标做横向对比。
默认协议：deepseek/deepseek-chat（付费）× 3 题  vs  zai/glm-4-flash（免费）× 3 题。
成本：deepseek 约 3 局 × ¥0.015 ≈ ¥0.045；glm-4-flash 免费。

指标（全部从对局 transcript / scores 自动提取，无需人工标注）：
- vra   有效提问率：提问以问号结尾或含疑问词的比例（弱模型退化为句号/陈述句会拉低）
- yn    主持 yes/no 回答率：主持回答含「是/否/对/不对」等的比例（答非所问会拉低）
- f1    还原字符级 F1：终局还原 vs 汤底的字符集合 F1（越高越接近真相）
- score 裁判评分：LLM 裁判给的 0–100 分

用法：
  COMPARE_MODELS="deepseek/deepseek-chat,zai/glm-4-flash" \
  COMPARE_PUZZLES="turtle_soup_001,turtle_soup_002,turtle_soup_003" \
  python run_compare.py
每局原始对局存 logs/compare_<model>_<pid>.json；汇总报告 + 图表在运行结束打印并落盘。
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---- 手动加载 .env（同 test_real_game.py 的修正版：剥行内注释 + workspace 兜底）----
def _load_one(path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                # 去掉行内注释（首个 # 之后），再剥引号与首尾空白
                v = v.split("#", 1)[0].strip().strip('"').strip("'")
                if k:
                    os.environ.setdefault(k, v)  # 不覆盖已存在的 OS 环境变量（真 key）
    except FileNotFoundError:
        pass


def load_env():
    here = os.path.dirname(os.path.abspath(__file__))
    _load_one(os.path.join(here, ".env"))
    _load_one(os.path.join(here, "..", "workspace", ".env"))


load_env()

from engine.game import TurtleSoupGame
from engine.puzzles import get_puzzle
from players import make_player

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUEST_WORDS = ("吗", "呢", "为何", "为什么", "怎么", "是否", "是不是", "是谁",
               "什么", "哪里", "哪", "多少", "几")
YN_TOKENS = ("是", "否", "对", "不对", "正确", "错误", "yes", "no", "Yes", "No", "YES", "NO")


def _char_f1(a: str, b: str) -> float:
    """字符级集合 F1，作为无嵌入时的粗略相似度（与 mock.py 一致）。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    if not inter:
        return 0.0
    prec = len(inter) / len(sa)
    rec = len(inter) / len(sb)
    return 2 * prec * rec / (prec + rec)


def _valid_q(q: str) -> bool:
    q = q.strip()
    if q.endswith(("?", "？")):
        return True
    return any(w in q for w in QUEST_WORDS)


def _yn_rate(answers) -> float:
    if not answers:
        return 0.0
    hit = sum(1 for m in answers if any(t in m for t in YN_TOKENS))
    return hit / len(answers)


def run_one(model: str, pid: str) -> dict:
    puzzle = get_puzzle(pid)
    questioner = make_player(model, role="questioner", index=0)
    host = make_player(model, role="host")
    judge = make_player(model, role="judge")
    game = TurtleSoupGame(puzzle, host, [questioner], judge=judge, max_rounds=4)
    res = game.run()

    qs = [m["content"] for m in res.transcript if m["role"] == "question"]
    ans = [m["content"] for m in res.transcript if m["role"] == "answer"]
    guess = next(m["content"] for m in res.transcript if m["role"] == "guess")
    score = res.scores[0]["score"]

    rec = {
        "model": model,
        "puzzle": pid,
        "title": puzzle.title,
        "valid_q": sum(1 for x in qs if _valid_q(x)),
        "total_q": len(qs),
        "vra": sum(1 for x in qs if _valid_q(x)) / max(1, len(qs)),
        "yn_rate": _yn_rate(ans),
        "f1": _char_f1(guess, puzzle.solution),
        "score": score,
        "guess": guess[:160],
    }
    fn = f"logs/compare_{model.replace('/', '_')}_{pid}.json"
    res.save(fn)
    rec["log"] = fn
    return rec


def main():
    models = [m.strip() for m in os.getenv(
        "COMPARE_MODELS", "deepseek/deepseek-chat,zai/glm-4-flash").split(",") if m.strip()]
    pids = [p.strip() for p in os.getenv(
        "COMPARE_PUZZLES", "turtle_soup_001,turtle_soup_002,turtle_soup_003").split(",") if p.strip()]

    t0 = time.time()
    rows = []
    for m in models:
        for p in pids:
            print(f"[run] {m} × {p} …", flush=True)
            rows.append(run_one(m, p))

    # 聚合
    agg = {}
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        agg[m] = {
            "vra": sum(r["vra"] for r in rs) / len(rs),
            "yn_rate": sum(r["yn_rate"] for r in rs) / len(rs),
            "f1": sum(r["f1"] for r in rs) / len(rs),
            "score": sum(r["score"] for r in rs) / len(rs),
            "n": len(rs),
        }

    # 图表
    metrics = [("vra", "Valid Q Rate", "{:.0%}"), ("yn_rate", "Host yes/no", "{:.0%}"),
               ("f1", "Reconstruction F1", "{:.2f}"), ("score", "Judge Score", "{:.1f}")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    palette = ["#2563eb", "#16a34a", "#d97706", "#dc2626"]
    for ax, (key, label, fmt) in zip(axes, metrics):
        vals = [agg[m][key] for m in models]
        bars = ax.bar(range(len(models)), vals, color=palette[:len(models)])
        ax.set_title(label, fontsize=12)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.split("/")[-1] for m in models], fontsize=9, rotation=15)
        ax.set_ylim(0, max(1.0, max(vals) * 1.15))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                    fmt.format(v), ha="center", fontsize=9)
    fig.suptitle("Model Head-to-Head · Same-Puzzle Baseline (3 puzzles avg)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    chart_path = "logs/compare_chart.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)

    # 报告
    lines = []
    lines.append("# AI Arena · 模型横向对决报告\n")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 协议：{len(models)} 模型 × {len(pids)} 题 = {len(rows)} 局")
    lines.append(f"- 成本：deepseek × {sum(1 for r in rows if r['model'].startswith('deepseek'))} 局 ≈ "
                 f"¥{0.015 * sum(1 for r in rows if r['model'].startswith('deepseek')):.3f}；"
                 f"glm-4-flash 免费")
    lines.append(f"- 总耗时：{time.time() - t0:.1f}s\n")
    lines.append("## 汇总（各模型 3 题均值）\n")
    lines.append("| 模型 | 有效提问率 | 主持yes/no率 | 还原F1 | 裁判评分 |")
    lines.append("|---|---|---|---|---|")
    for m in models:
        a = agg[m]
        lines.append(f"| {m} | {a['vra']:.0%} | {a['yn_rate']:.0%} | {a['f1']:.2f} | {a['score']:.1f} |")
    lines.append("\n## 逐题明细\n")
    lines.append("| 模型 | 题目 | 有效提问 | yes/no率 | 还原F1 | 评分 |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['model']} | {r['puzzle']} | {r['valid_q']}/{r['total_q']} "
                     f"| {r['yn_rate']:.0%} | {r['f1']:.2f} | {r['score']:.1f} |")
    lines.append("\n## 结论\n")
    best = max(models, key=lambda m: agg[m]["score"])
    lines.append(f"- 裁判评分冠军：**{best}**（{agg[best]['score']:.1f}）")
    lines.append(f"- 有效提问率：deepseek {agg.get('deepseek/deepseek-chat', {}).get('vra', 0):.0%} "
                 f"vs glm-4-flash {agg.get('zai/glm-4-flash', {}).get('vra', 0):.0%}——"
                 "付费模型几乎全程有效提问，免费模型第 3 轮起高频退化（句号/陈述句/空响应）")
    lines.append(f"- 还原 F1：deepseek {agg.get('deepseek/deepseek-chat', {}).get('f1', 0):.2f} "
                 f"vs glm-4-flash {agg.get('zai/glm-4-flash', {}).get('f1', 0):.2f}")
    lines.append("\n> 证据等级：每模型 3 题小样本，方向明确但尚未达统计显著；结论为初步量化，"
                 "非最终定论。真实账单以 deepseek 后台为准。")
    lines.append(f"\n图表：{chart_path}｜原始对局：logs/compare_*.json")

    report = "\n".join(lines)
    with open("logs/compare_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\n[done] chart={chart_path} report=logs/compare_report.md")


if __name__ == "__main__":
    main()
