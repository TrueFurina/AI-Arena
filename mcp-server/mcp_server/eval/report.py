"""评测报告（v0.1 基础版）：图表 + 文字结论。

生成一张"各模型胜场 / 平均最佳猜想命中率"柱状图，并附可读文字结论，
方便直接发小红书 / 比赛展示。v0.3 会扩展为多维度能力雷达图。
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt  # noqa: E402

from mcp_server.eval.metrics import compute_leaderboard, compute_profile  # noqa: E402

# 雷达图的四个维度及其中文标签（与 metrics._PROFILE_DIMS 对应）
_RADAR_DIMS = ("accuracy", "efficiency", "breadth", "decisiveness")
_RADAR_LABELS = ("推理准确", "信息效率", "命中广度", "收网果断")

# 尽力注册一个可用的中文字体，使图表中文标签能正常渲染；
# 若环境中没有中文字体，则图表退化为英文标签（见 generate_chart），不影响功能。
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]
for _cjk in _CJK_CANDIDATES:
    if os.path.exists(_cjk):
        try:
            fm.fontManager.addfont(_cjk)
            _name = fm.FontProperties(fname=_cjk).get_name()
            plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:  # noqa: BLE001
            pass


def _is_empty(board: object) -> bool:
    if board is None:
        return True
    return bool(getattr(board, "empty", True))  # DataFrame.empty 是属性


def build_report_text(board: "object", profile: "object" = None) -> str:
    """根据排行榜与能力画像生成文字结论。"""
    if _is_empty(board):
        return "暂无对局数据，先跑几局 `run_match` 再来生成报告吧。"
    lines = ["# AI 角斗场 · 评测速览", ""]
    for _, r in board.iterrows():
        lines.append(
            f"- **{r['model']}**：对局 {int(r['games'])} 场，胜 {int(r['wins'])} 场，"
            f"胜率 {r['win_rate']:.0%}，平均猜想命中率 {r['avg_best_guess']:.0%}"
        )
    top = board.iloc[0]
    lines += ["", f"🏆 当前领跑：**{top['model']}**（胜率 {top['win_rate']:.0%}）"]

    if profile is not None and not _is_empty(profile):
        lines += [
            "",
            "## 推理能力画像（四维）",
            "",
            "| 玩家 | 推理准确 | 信息效率 | 命中广度 | 收网果断 |",
            "|---|---|---|---|---|",
        ]
        for _, r in profile.iterrows():
            lines.append(
                f"| {r['model']} | {r['accuracy']:.0%} | {r['efficiency']:.2f} "
                f"| {r['breadth']:.0%} | {r['decisiveness']:.0%} |"
            )
        lines += [
            "",
            "> 说明：项目目标是推理/欺骗/伪装/辩论四维能力，但后三者需对抗模式才能观测，",
            "> 当前雷达图只反映**推理侧的四种风格**，不是完整的四维能力评分。",
        ]
    return "\n".join(lines)


def generate_chart(board: "object", output_path: str | Path) -> str:
    """生成柱状图（胜场 + 平均命中率双轴），保存为 PNG，返回路径。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_empty(board):
        # 空图占位
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "暂无数据", ha="center", va="center", fontsize=16)
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return str(output_path)

    models = board["model"].tolist()
    x = range(len(models))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(x, board["wins"], color="#4C72B0", alpha=0.8, label="胜场")
    ax1.set_ylabel("胜场", color="#4C72B0")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(models, rotation=15)

    ax2 = ax1.twinx()
    ax2.plot(x, board["avg_best_guess"], color="#DD8452", marker="o", label="平均猜想命中率")
    ax2.set_ylabel("平均猜想命中率", color="#DD8452")
    ax2.set_ylim(0, 1)

    fig.suptitle("AI 角斗场 · 模型表现")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return str(output_path)


def generate_radar(profile: "object", output_path: str | Path) -> str:
    """生成四维推理画像雷达图，保存为 PNG，返回路径。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_empty(profile):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.text(0.5, 0.5, "暂无数据", ha="center", va="center", fontsize=16)
        ax.axis("off")
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return str(output_path)

    n = len(_RADAR_DIMS)
    angles = [i / n * 2 * math.pi for i in range(n)]
    angles += angles[:1]  # 闭合折线

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"polar": True})
    for _, row in profile.iterrows():
        vals = [float(row[d]) for d in _RADAR_DIMS]
        vals += vals[:1]
        ax.plot(angles, vals, "-o", linewidth=2, label=str(row["model"]))
        ax.fill(angles, vals, alpha=0.12)

    ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], _RADAR_LABELS)
    ax.set_ylim(0, 1)
    ax.set_title("AI 角斗场 · 推理能力画像", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def export_report(output_dir: str | Path, exclude_cheating: bool = True) -> dict[str, str]:
    """导出报告：排行榜图 + 能力雷达图 + 文字结论 MD，返回路径字典。

    exclude_cheating 默认 True：作弊玩家（mock 预言机）不参与排名与画像，
    否则它每问必中、效率满分，会把雷达图撑爆且结论毫无意义。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    board = compute_leaderboard(exclude_cheating=exclude_cheating)
    profile = compute_profile(exclude_cheating=exclude_cheating)
    chart = generate_chart(board, output_dir / "leaderboard.png")
    radar = generate_radar(profile, output_dir / "radar.png")
    text = build_report_text(board, profile)
    md_path = output_dir / "report.md"
    md_path.write_text(text, encoding="utf-8")
    return {"chart": chart, "radar": radar, "markdown": str(md_path), "text": text}
