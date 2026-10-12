"""推理指标模块（M2）。对应 specs/engine-design.md §7 与 EVALUATION.md。

设计原则：
- 只读「对局 JSON 日志」，不依赖 engine，可独立复算。
- 还原度直接信任日志里的 result.restoration（由 Game 关键词感知判定产出）；
  本模块额外提供 guess_score（汤底保真度）与 efficiency（单位轮次还原度），
  并支持跨多局按模型聚合，用于横向比较不同 LLM 的推理能力。
- pandas / matplotlib 为可选依赖：缺失时退化为纯 Python 字典/列表，仍可产出文字结论。
"""

import glob
import json
import os


# ---- 基础指标 ----

def guess_score(declared, bottom):
    """汤底保真度：声明汤底与真实汤底的二元组 Jaccard 重合度（0~1）。"""
    if not declared:
        return 0.0

    def _bigrams(s):
        s = "".join(ch for ch in s if ch.strip())
        return set(s[i:i + 2] for i in range(len(s) - 1))

    b1, b2 = _bigrams(declared), _bigrams(bottom)
    if not b2:
        return 0.0
    return len(b1 & b2) / len(b2)


def metrics_from_log(log):
    """从单份日志抽取指标行。"""
    r = log.get("result", {})
    meta = log.get("meta", {})
    deception = r.get("deception", {}) or {}
    return {
        "model": meta.get("model", "unknown"),
        "puzzle_id": meta.get("puzzle_id", "unknown"),
        "restoration": r.get("restoration", 0.0),
        "guess_score": r.get("guess_score", 0.0),
        "efficiency": r.get("efficiency", 0.0),
        "turns_used": r.get("turns_used", 0),
        "declared": r.get("declared", False),
        "n_facts": len(log.get("key_facts", []) or []),
        "n_hit": len(r.get("key_facts_hit", []) or []),
        # M5 欺骗维度（向后兼容：旧日志缺 deception 时记 0）
        "lies_total": deception.get("lies_total", 0),
        "lie_rate": deception.get("lie_rate", 0.0),
    }


# ---- 读取与聚合 ----

def load_logs(log_dir):
    """读取目录下所有对局 JSON，返回日志字典列表（解析失败的跳过）。"""
    logs = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                logs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return logs


def summarize(logs):
    """返回每份日志的指标行列表。"""
    return [metrics_from_log(log) for log in logs]


def aggregate_by_model(rows):
    """按模型聚合：均值还原度 / guess / efficiency / 轮次，以及样本数。"""
    buckets = {}
    for row in rows:
        buckets.setdefault(row["model"], []).append(row)
    agg = {}
    for model, rs in buckets.items():
        n = len(rs)
        agg[model] = {
            "n": n,
            "mean_restoration": round(sum(r["restoration"] for r in rs) / n, 4),
            "mean_guess": round(sum(r["guess_score"] for r in rs) / n, 4),
            "mean_efficiency": round(sum(r["efficiency"] for r in rs) / n, 4),
            "mean_turns": round(sum(r["turns_used"] for r in rs) / n, 2),
            "declared_rate": round(sum(1 for r in rs if r["declared"]) / n, 4),
            # M5 欺骗维度
            "mean_lie_rate": round(sum(r.get("lie_rate", 0.0) for r in rs) / n, 4),
        }
    return agg


# ---- M5 欺骗维度度量 ----

def mean_lie_rate(rows):
    """所有样本的实际撒谎率均值（GM 撒谎比例，与提问者无关）。"""
    if not rows:
        return 0.0
    return round(sum(r.get("lie_rate", 0.0) for r in rows) / len(rows), 4)


def deception_robustness(restoration_truth, restoration_lie):
    """欺骗鲁棒性 ∈ [0,1]：在撒谎环境下还原度相对「诚实环境」的保持比例。

    = 1 表示 GM 撒谎完全不影响推理；= 0 表示一撒谎就崩溃。
    分母为 0 时（诚实环境本就还原不出）返回 0，避免虚高。
    """
    if restoration_truth <= 0:
        return 0.0
    return round(max(0.0, min(1.0, restoration_lie / restoration_truth)), 4)


# ---- 报告产出 ----

def _setup_cjk_font():
    """尝试注册系统中的 CJK 字体（如 Noto Sans CJK），让图表中文不乱码。
    找不到则返回 None（调用方退化为英文标签）。"""
    try:
        import glob
        import matplotlib
        from matplotlib import font_manager
    except ImportError:
        return None
    candidates = []
    for pat in ("/usr/share/fonts/opentype/noto/*CJK*",
                "/usr/share/fonts/truetype/noto/*CJK*",
                "/usr/share/fonts/**/*CJK*.ttc",
                "/usr/share/fonts/**/*CJK*.ttf"):
        candidates += glob.glob(pat, recursive=True)
    for f in candidates:
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            continue
    cjk_names = [n.name for n in font_manager.fontManager.ttflist
                 if "CJK" in n.name]
    if cjk_names:
        matplotlib.rcParams["font.sans-serif"] = [cjk_names[0]] + list(matplotlib.rcParams["font.sans-serif"])
        matplotlib.rcParams["axes.unicode_minus"] = False
        return cjk_names[0]
    return None


def make_chart(agg, out_path):
    """按模型画「均值还原度」柱状图；无 matplotlib 时返回 None（文字结论仍可用）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    cjk = _setup_cjk_font()

    models = list(agg.keys())
    values = [agg[m]["mean_restoration"] for m in models]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color="#4C72B0")
    ax.set_ylim(0, 1)
    # 无中文字体时退化为英文标签，避免方块乱码
    ax.set_ylabel("均值还原度 (restoration)" if cjk else "Mean Restoration")
    ax.set_title("AI-Arena · 各模型海龟汤推理还原度" if cjk
                 else "AI-Arena · Turtle Soup Reasoning Restoration by Model")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def text_report(agg):
    """生成可读的中文文字结论（规则 5：图表 + 文字）。"""
    if not agg:
        return "（暂无对局数据，无法生成结论）"
    ranked = sorted(agg.items(), key=lambda kv: kv[1]["mean_restoration"], reverse=True)
    lines = ["【AI-Arena 推理评测结论】", ""]
    lines.append(f"参与模型 {len(ranked)} 个，按均值还原度排序：")
    for i, (model, m) in enumerate(ranked, 1):
        lines.append(
            f"{i}. {model} — 还原度 {m['mean_restoration']:.2f} | "
            f"汤底保真 {m['mean_guess']:.2f} | 效率 {m['mean_efficiency']:.3f} | "
            f"平均 {m['mean_turns']:.1f} 轮 | 样本 {m['n']}"
        )
    best = ranked[0][0]
    worst = ranked[-1][0]
    lines.append("")
    if best != worst:
        lines.append(f"结论：{best} 推理还原度最高，{worst} 最低。"
                     f"还原度衡量「用最少是非题探测到多少关键事实」，数值越高代表推理/提问策略越优。")
    else:
        lines.append("结论：当前仅单一模型样本，建议补充多模型对局以横向比较。")
    return "\n".join(lines)
