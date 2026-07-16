"""评测报告（v0.1 基础版）：图表 + 文字结论。

生成一张"各模型胜场 / 平均最佳猜想命中率"柱状图，并附可读文字结论，
方便直接发小红书 / 比赛展示。v0.3 会扩展为多维度能力雷达图。
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt  # noqa: E402

from mcp_server.eval.metrics import compute_leaderboard  # noqa: E402

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


def build_report_text(board: "object") -> str:
    """根据排行榜生成文字结论。"""
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


def export_report(output_dir: str | Path) -> dict[str, str]:
    """导出报告：图表 PNG + 文字结论 MD，返回文件路径字典。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    board = compute_leaderboard()
    chart = generate_chart(board, output_dir / "leaderboard.png")
    text = build_report_text(board)
    md_path = output_dir / "report.md"
    md_path.write_text(text, encoding="utf-8")
    return {"chart": chart, "markdown": str(md_path), "text": text}
