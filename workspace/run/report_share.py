"""M4 可读战报生成器：把 metrics.csv 变成可发小红书 / 可投稿比赛的报告。

为什么存在（规则 5）：
- tournament / evaluate 产出的是工程态产物（CSV / TXT / PNG），人读不直观。
- 本脚本把它们「翻译」成一份带战力榜、可视化、各题矩阵、以及
  现成文案提示的 Markdown，直接贴小红书或塞进比赛报告。

用法：
    # 默认读 analysis/metrics.csv，写 analysis/SHARE.md + 刷新 analysis/restoration.png
    python run/report_share.py
    python run/report_share.py --csv analysis/metrics.csv --out analysis/SHARE.md

不依赖 API key，纯离线复算。pandas 缺失时退化为标准库 csv。

设计纪律（见 doc/METHODOLOGY.md）：
- 零 key 优先：本脚本只读 CSV，绝不触发任何 LLM 调用。
- 配置/代码分离：模型名、指标阈值都来自数据，脚本只负责排版。
- 先度量后优化：所有数字都来自 tournament 真实落盘的指标，不编造。
"""

import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import metrics  # 复用聚合与绘图能力


# ---------------------------------------------------------------------------
# 数据读取（优先标准库，pandas 只是锦上添花）
# ---------------------------------------------------------------------------

def read_rows(csv_path):
    """读 metrics.csv，把数值列转成 float/int，返回行字典列表。"""
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            for num_col in ("restoration", "guess_score", "efficiency"):
                row[num_col] = float(row.get(num_col) or 0.0)
            for int_col in ("turns_used", "n_facts", "n_hit"):
                row[int_col] = int(float(row.get(int_col) or 0))
            row["declared"] = str(row.get("declared", "")).lower() in ("true", "1", "yes")
            rows.append(row)
    return rows


def puzzle_pivot(rows):
    """构造 模型 × 题目 的均值还原度矩阵，供「各题表现」表使用。"""
    bucket = {}  # (model, puzzle) -> [restoration...]
    for r in rows:
        bucket.setdefault((r["model"], r["puzzle_id"]), []).append(r["restoration"])
    puzzles = sorted({r["puzzle_id"] for r in rows})
    models = sorted({r["model"] for r in rows})
    pivot = {}
    for m in models:
        pivot[m] = {p: (sum(bucket[(m, p)]) / len(bucket[(m, p)]) if (m, p) in bucket else None)
                    for p in puzzles}
    return models, puzzles, pivot


# ---------------------------------------------------------------------------
# Markdown 排版
# ---------------------------------------------------------------------------

def _rank_medal(i):
    return {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, f"{i + 1}️⃣")


def build_markdown(csv_path, agg, models, puzzles, pivot, chart_rel, generated_at):
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["mean_restoration"], reverse=True)
    best, worst = ranked[0][0], ranked[-1][0]

    # 标题与摘要
    md = []
    md.append("# 🤖 AI-Arena · 海龟汤推理擂台战报")
    md.append("")
    md.append(f"> 自动生成 · `{os.path.basename(csv_path)}` · {generated_at}")
    md.append(">")
    md.append("> AI-vs-AI 离线社交推理评测 · 用最少的是非题，探测出多少关键事实")
    md.append("")

    # 战力榜
    md.append("## 🏆 模型战力榜")
    md.append("")
    md.append("| 排名 | 模型 | 还原度 | 汤底保真 | 效率 | 平均轮次 | 样本 |")
    md.append("|------|------|--------|----------|------|----------|------|")
    for i, (model, m) in enumerate(ranked):
        md.append(
            f"| {_rank_medal(i)} | {model} | {m['mean_restoration']:.2f} | "
            f"{m['mean_guess']:.2f} | {m['mean_efficiency']:.3f} | "
            f"{m['mean_turns']:.1f} | {m['n']} |"
        )
    md.append("")

    # 指标速读
    md.append("<details><summary>📖 指标怎么读（点开）</summary>")
    md.append("")
    md.append("- **还原度 (restoration)**：被 GM 确认的关键事实 / 关键事实总数，越高越会问。")
    md.append("- **汤底保真 (guess_score)**：最终猜的汤底与真相的二元组重合度，越高越准。")
    md.append("- **效率 (efficiency)**：`还原度 ÷ 轮次`，单位轮次的推理性价比。")
    md.append("- **平均轮次**：达到结论平均用多少轮提问。")
    md.append("</details>")
    md.append("")

    # 一句话结论
    md.append("## 📌 一句话结论")
    md.append("")
    if best != worst:
        md.append(
            f"**{best}** 推理还原度最高（{agg[best]['mean_restoration']:.2f}），"
            f"**{worst}** 最低（{agg[worst]['mean_restoration']:.2f}）。"
            f"还原度衡量「用最少是非题探测到多少关键事实」，数值越高代表提问/推理策略越优。"
        )
    else:
        md.append("当前仅单一模型样本，建议补充多模型对局以横向比较战力。")
    md.append("")

    # 可视化
    md.append("## 📊 可视化")
    md.append("")
    md.append(f"![各模型均值还原度]({chart_rel})")
    md.append("")

    # 各题表现矩阵
    md.append("## 🧩 各题表现（均值还原度）")
    md.append("")
    header = "| 模型 | " + " | ".join(puzzles) + " | 均值 |"
    sep = "|------|" + "|".join(["------"] * len(puzzles)) + "|------|"
    md.append(header)
    md.append(sep)
    for m in models:
        cells = []
        vals = []
        for p in puzzles:
            v = pivot[m][p]
            if v is None:
                cells.append("—")
            else:
                cells.append(f"{v:.2f}")
                vals.append(v)
        mean = sum(vals) / len(vals) if vals else 0.0
        md.append(f"| {m} | " + " | ".join(cells) + f" | {mean:.2f} |")
    md.append("")
    md.append("> 单元格为空（—）代表该题无该模型样本；行末均值跨所有有样本的题目。")
    md.append("")

    # 小红书文案提示
    md.append("## 📱 小红书文案提示（可直接改）")
    md.append("")
    title_best = ranked[0][0]
    md.append(f"- **标题候选**：「{len(ranked)}个AI比拼海龟汤，{title_best} 用最少问题猜出真相🔥」")
    md.append("- **正文钩子**：「当 AI 互相出题互相问，谁的逻辑更强？我们让大模型玩起了海龟汤……」")
    md.append("- **数据落点**：贴上面的战力榜 + 还原度图，强调「还原度」这个反直觉指标。")
    md.append("- **话题标签**：`#AI` `#大模型评测` `#海龟汤` `#AI推理` `#Agent对战` `#AIGC`")
    md.append("- **互动尾句**：「你觉得哪个模型最会问问题？评论区聊聊👀」")
    md.append("")

    # 比赛文案提示
    md.append("## 🏅 比赛 / 技术报告文案提示")
    md.append("")
    md.append("- **问题定义**：在离线、零人工干预下，以结构化 JSON 动作协议驱动 AI-vs-AI 海龟汤对局，")
    md.append("  用「最少轮次内还原关键事实的比例」量化模型提问与推理能力。")
    md.append("- **方法亮点**：关键词感知 GM（SmartGM）做可复算判定；litellm 统一接入多供应商；")
    md.append("  TOML 模型注册表 + 零 key 冒烟，保证可复现与可审计。")
    md.append("- **结论句式**：「在 N 个模型、M 局对局中，X 以 A 还原度领先；效率指标显示……」")
    md.append("- **可复现声明**：附仓库地址与 `python run/tournament.py --mock` 一键复现命令。")
    md.append("")

    md.append("---")
    md.append(f"_由 `run/report_share.py` 生成 · 数据源 `{csv_path}`_")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="M4 可读战报生成器")
    parser.add_argument("--csv", default="analysis/metrics.csv", help="指标明细 CSV")
    parser.add_argument("--out", default="analysis/SHARE.md", help="输出 Markdown 路径")
    parser.add_argument("--chart", default="analysis/restoration.png", help="图表输出路径")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"❌ 找不到 {args.csv}，请先跑 run/tournament.py 或 run/evaluate.py 生成指标。")

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit(f"❌ {args.csv} 为空，无可聚合样本。")

    agg = metrics.aggregate_by_model(rows)
    models, puzzles, pivot = puzzle_pivot(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 刷新图表（复用 metrics.make_chart，无 matplotlib 时图表留空引用）
    chart_rel = os.path.relpath(args.chart, os.path.dirname(os.path.abspath(args.out))) or "restoration.png"
    chart = metrics.make_chart(agg, args.chart)
    if chart is None:
        chart_rel = "restoration.png（未安装 matplotlib，已跳过）"

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = build_markdown(args.csv, agg, models, puzzles, pivot, chart_rel, generated_at)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    print("✅ 战报已生成")
    print(f"  输出 : {args.out}")
    print(f"  图表 : {args.chart if chart else chart_rel}")
    print(f"  模型 : {len(agg)} 个 | 题目 : {len(puzzles)} 道 | 样本 : {len(rows)} 局")


if __name__ == "__main__":
    main()
