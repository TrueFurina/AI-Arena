#!/usr/bin/env python3
"""
AI Arena · 双模型同局交叉对决（cross-play matrix）
====================================================
把「提问者」与「主持」两个角色拆开，由不同模型担任，构造 2×2 角色矩阵，
隔离『提问者能力』与『主持质量』对终局还原度的独立贡献。

为什么需要这个实验：
  B 方案（run_compare.py）让同一模型同时担任提问者+主持，得到的是「模型综合得分」，
  无法区分「提问者强」还是「主持强」。cross-play 把两角色解耦后，可以做 2×2 析因：
    - 固定提问者看主持效应：F1(DD) vs F1(DG)（强提问者被弱主持拖累多少）
    - 固定主持看提问者效应：F1(DD) vs F1(GD)（强主持能否救起弱提问者）
    - 交互效应：两角色是否存在协同/拮抗

协议（成本可控，与 B 对齐：max_rounds=4）：
- 模型：deepseek/deepseek-chat（付费）× zai/glm-4-flash（免费）
- 4 个角色配置 × 3 题：
    DD = deepseek 提问 + deepseek 主持（对角，来自 B 历史日志）
    GG = glm 提问 + glm 主持     （对角，来自 B 历史日志）
    DG = deepseek 提问 + glm 主持（交叉，本次新跑）
    GD = glm 提问 + deepseek 主持 （交叉，本次新跑）
- 不接 LLM 裁判：以字符级 F1（还原 vs 汤底）为客观主指标，避免裁判偏差污染对比。
- 成本：本次仅跑 DG+GD（deepseek 出现在 6 局）≈ ¥0.09；DD/GG 复用 B 日志零新增成本。

指标（全部自动提取，无人工标注）：
- vra   有效提问率：提问以问号结尾或含疑问词的比例
- yn    主持 yes/no 回答率
- f1    还原字符级 F1（客观，主指标）

用法：
  python run_crossplay.py
"""
import os
import sys
import json
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---- 手动加载 .env（与 run_compare.py / test_real_game.py 一致的修正版）----
def _load_one(path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
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

DEEPSEEK = "deepseek/deepseek-chat"
GLM = "zai/glm-4-flash"
MODEL_SHORT = {DEEPSEEK: "deepseek", GLM: "glm-4-flash"}


def _char_f1(a: str, b: str) -> float:
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


def metrics_from_result(res: dict) -> dict:
    """从 GameResult dict 提取指标（提问/主持/还原 分离）。"""
    tr = res["transcript"]
    qs = [m["content"] for m in tr if m["role"] == "question"]
    ans = [m["content"] for m in tr if m["role"] == "answer"]
    guess = next((m["content"] for m in tr if m["role"] == "guess"), "")
    solution = res["puzzle"].get("solution", "")
    vq = sum(1 for x in qs if _valid_q(x))
    return {
        "valid_q": vq,
        "total_q": len(qs),
        "vra": vq / max(1, len(qs)),
        "yn_rate": _yn_rate(ans),
        "f1": _char_f1(guess, solution),
        "guess": guess[:200],
    }


def run_cell(q_model: str, h_model: str, pid: str) -> dict:
    puzzle = get_puzzle(pid)
    questioner = make_player(q_model, role="questioner", index=0)
    host = make_player(h_model, role="host")
    game = TurtleSoupGame(puzzle, host, [questioner], judge=None, max_rounds=4)
    res = game.run()
    rec = metrics_from_result(res.to_dict())
    rec.update({"q_model": q_model, "h_model": h_model, "puzzle": pid,
                "title": puzzle.title})
    fn = f"logs/cross_{MODEL_SHORT[q_model]}_Q_{MODEL_SHORT[h_model]}_H_{pid}.json"
    res.save(fn)
    rec["log"] = fn
    return rec


def diag_from_logs(pattern: str, q_model: str, h_model: str) -> list:
    """复用 B 方案的对角日志（同 max_rounds=4 协议，F1 口径一致）。"""
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        rec = metrics_from_result(res)
        rec.update({"q_model": q_model, "h_model": h_model,
                    "puzzle": res["puzzle"].get("soup_id", ""),
                    "title": res["puzzle"].get("title", ""), "log": path})
        out.append(rec)
    return out


def main():
    pids = ["turtle_soup_001", "turtle_soup_002", "turtle_soup_003"]
    t0 = time.time()
    cells = {}

    # DD / GG 对角：复用 B 历史日志（零新增成本）
    cells["DD"] = diag_from_logs("logs/compare_deepseek_deepseek-chat_*.json",
                                 DEEPSEEK, DEEPSEEK)
    cells["GG"] = diag_from_logs("logs/compare_zai_glm-4-flash_*.json", GLM, GLM)

    # DG / GD 交叉：本次新跑（deepseek 付费，glm 免费）
    cells["DG"] = [run_cell(DEEPSEEK, GLM, p) for p in pids]
    cells["GD"] = [run_cell(GLM, DEEPSEEK, p) for p in pids]

    # 校验样本完整性
    for k, v in cells.items():
        if len(v) != len(pids):
            print(f"[warn] 配置 {k} 仅 {len(v)} 题（期望 {len(pids)}），可能缺 B 日志")

    # 聚合（3 题均值）
    agg = {}
    for k, rows in cells.items():
        agg[k] = {
            "vra": sum(r["vra"] for r in rows) / len(rows),
            "yn_rate": sum(r["yn_rate"] for r in rows) / len(rows),
            "f1": sum(r["f1"] for r in rows) / len(rows),
            "n": len(rows),
        }

    # ---- 图表：2×2 F1 矩阵（双面板：F1 + vra）----
    hosts = [DEEPSEEK, GLM]
    qers = [DEEPSEEK, GLM]
    label_of = {DEEPSEEK: "deepseek", GLM: "glm-4-flash"}
    cell_of = {(DEEPSEEK, DEEPSEEK): "DD", (GLM, GLM): "GG",
               (DEEPSEEK, GLM): "DG", (GLM, DEEPSEEK): "GD"}
    palette = {"deepseek": "#2563eb", "glm-4-flash": "#16a34a"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (metric, title, fmt) in zip(
            axes, [("f1", "Reconstruction F1 (objective)", "{:.2f}"),
                   ("vra", "Valid Question Rate", "{:.0%}")]):
        # x 轴=主持，分组色=提问者
        x = range(len(hosts))
        w = 0.35
        for i, qm in enumerate(qers):
            vals = []
            for hm in hosts:
                key = cell_of[(qm, hm)]
                vals.append(agg[key][metric])
            bars = ax.bar([xi + (i - 0.5) * w for xi in x], vals, width=w,
                          color=palette[label_of[qm]], label=f"Q={label_of[qm]}")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                        fmt.format(v), ha="center", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"Host={label_of[h]}" for h in hosts], fontsize=10)
        ax.set_ylim(0, max(1.0, max(agg[k][metric] for k in agg) * 1.18))
        ax.legend(fontsize=9)
    fig.suptitle("Cross-Play 2x2 Matrix · Questioner x Host (3 puzzles avg)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    chart_path = "logs/crossplay_chart.png"
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)

    # ---- 报告 ----
    lines = []
    lines.append("# AI Arena · 双模型同局交叉对决报告（Cross-Play 2×2）\n")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 协议：4 角色配置 × 3 题；max_rounds=4；无 LLM 裁判（F1 客观主指标）")
    lines.append(f"- 成本：本次新跑 DG+GD 共 6 局（deepseek 付费）≈ ¥0.09；"
                 f"DD/GG 复用 B 日志零新增成本")
    lines.append(f"- 总耗时：{time.time() - t0:.1f}s\n")

    lines.append("## 2×2 还原 F1 矩阵（3 题均值，越高越接近真相）\n")
    lines.append("| 提问者 \\ 主持 | deepseek | glm-4-flash |")
    lines.append("|---|---|---|")
    for qm in qers:
        row = [f"| {label_of[qm]} "]
        for hm in hosts:
            row.append(f"| {agg[cell_of[(qm, hm)]]['f1']:.2f} ")
        lines.append("".join(row) + "|")
    lines.append("\n> 读法：行=提问者，列=主持。DD=自对弈深模，GG=自对弈免费模，"
                 "DG=深提问+免主持，GD=免提问+深主持。\n")

    lines.append("## 完整指标汇总\n")
    lines.append("| 配置 | 提问者 | 主持 | 有效提问率 | 主持yes/no率 | 还原F1 |")
    lines.append("|---|---|---|---|---|---|")
    for k in ["DD", "GG", "DG", "GD"]:
        a = agg[k]
        lines.append(f"| {k} | {label_of[cells[k][0]['q_model']]} | "
                     f"{label_of[cells[k][0]['h_model']]} | "
                     f"{a['vra']:.0%} | {a['yn_rate']:.0%} | {a['f1']:.2f} |")

    lines.append("\n## 逐题明细\n")
    lines.append("| 配置 | 题目 | 有效提问 | yes/no率 | 还原F1 |")
    lines.append("|---|---|---|---|---|")
    for k in ["DD", "GG", "DG", "GD"]:
        for r in cells[k]:
            lines.append(f"| {k} | {r['puzzle']} | {r['valid_q']}/{r['total_q']} "
                         f"| {r['yn_rate']:.0%} | {r['f1']:.2f} |")

    lines.append("\n## 析因结论\n")
    # 主持效应（固定提问者）
    host_eff_ds = agg["DD"]["f1"] - agg["DG"]["f1"]   # deepseek 提问：强主持-弱主持
    host_eff_glm = agg["GG"]["f1"] - agg["GD"]["f1"]  # glm 提问：弱主持-强主持
    # 提问者效应（固定主持）
    q_eff_ds = agg["DD"]["f1"] - agg["GD"]["f1"]      # deepseek 主持：强提问-弱提问
    q_eff_glm = agg["DG"]["f1"] - agg["GG"]["f1"]     # glm 主持：强提问-弱提问
    lines.append(f"- **主持效应**：固定深提问者，把主持从 deepseek 换成 glm，"
                 f"F1 变化 {host_eff_ds:+.2f}；固定免提问者，把主持从 glm 换成 deepseek，"
                 f"F1 变化 {host_eff_glm:+.2f}。")
    lines.append(f"- **提问者效应**：固定深主持，把提问从 glm 换成 deepseek，"
                 f"F1 变化 {q_eff_ds:+.2f}；固定免主持，把提问从 glm 换成 deepseek，"
                 f"F1 变化 {q_eff_glm:+.2f}。")
    lines.append(f"- **主导因子**：{'提问者' if abs(q_eff_ds) >= abs(host_eff_ds) else '主持'}"
                 f"对终局还原度的影响更大（基于 deepseek 行对比）。")
    lines.append(f"- 交互检验：DD-GG（对角差）={agg['DD']['f1']-agg['GG']['f1']:+.2f}，"
                 f"DG-GD（交叉差）={agg['DG']['f1']-agg['GD']['f1']:+.2f}；"
                 "若两差接近，说明角色近似可加、无强协同/拮抗。")
    lines.append("\n> 证据等级：每配置 3 题小样本，方向参考；F1 为字符级集合相似度，"
                 "对长文本还原偏乐观，仅作相对排序用，非绝对还原度。真实账单以 deepseek 后台为准。")
    lines.append(f"\n图表：{chart_path}｜原始对局：logs/cross_*.json")

    report = "\n".join(lines)
    with open("logs/crossplay_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\n[done] chart={chart_path} report=logs/crossplay_report.md")


if __name__ == "__main__":
    main()
