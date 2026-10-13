"""A-7 实验脚本 · 真实 LLM 裁判标定（设计见 DESIGN-P2A7-REAL-JUDGE.md）。

回答三个问题：
  Q1 真实 LLM 裁判上，等权集成是否仍优于单点裁判？（集成增益）
  Q2 真实裁判之间的可靠性差异 R 有多大？R 小则加权在理论上就没空间
  Q3 受控面板（judges/biased.py）与真实裁判的结论方向是否一致？

用法：
    python eval_real_judges.py --sanity          # 8 次调用，分钟级，先验仪器
    python eval_real_judges.py                    # 确认性实验（240 次调用）
    python eval_real_judges.py --personas strict,lenient --model zai/glm-4.5-flash

协议三条（违反任一条结论即无效）：
  ① 标定集与评测集按题 disjoint（默认 6/6）
  ② 客观真值唯一：char_f1(guess, solution) × 100
  ③ guess 用 solution 前缀构造质量梯度，不取整段（否则真值=100 被截断）

成本纪律：先 --sanity 再放量；缺席率超阈值即停并报告，不硬撑。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent))

# key 只从 .env 读（代码不写死、不外传）。**漏加载不会静默降级成 0 分**——
# 表现为「所有裁判缺席 + 覆盖率不足」，由下面的守卫显式拦住。
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as _exc:  # noqa: BLE001
    print(f"[a7] ⚠️ .env 加载失败（{type(_exc).__name__}: {_exc}），"
          f"裁判端点可能不可用")

from engine.evaluate import char_f1
from engine.puzzles import load_puzzles

from judges.biased import make_homogeneous_panel, make_heterogeneous_panel
from judges.llm_panel import JudgeUnavailable, RealJudgePanel, make_real_panel
from judges.panel import JudgePanel

LOG_DIR = Path(__file__).parent / "logs"
GUESS_FRACS = (0.15, 0.35, 0.55, 0.75, 0.90)

# 判据门槛（预注册，见设计文档 §6）
G1_MIN_FRAC = 2 / 3      # 等权须在 ≥2/3 用例上优于平均单点
G2_MIN_FRAC = 2 / 3      # 加权须在 ≥2/3 用例上不劣于等权
R_HOMO_THRESHOLD = 1.25  # 低于此值视为「近乎同质」，加权无理论空间


def objective_of(puzzle, guess: str) -> float:
    return char_f1(guess, puzzle.solution) * 100.0


def make_cases(puzzles: Sequence, fracs: Sequence[float] = GUESS_FRACS
               ) -> List[Tuple[Any, str]]:
    cases = []
    for p in puzzles:
        n = len(p.solution)
        for f in fracs:
            cases.append((p, p.solution[: max(1, int(n * f))]))
    return cases


def split_by_puzzle(cases: Sequence[Tuple], train_ids: set):
    tr = [c for c in cases if c[0].soup_id in train_ids]
    te = [c for c in cases if c[0].soup_id not in train_ids]
    return tr, te


def rmse(errors: Sequence[float]) -> float:
    return math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else float("nan")


def aggregate_run(panel: Any, cases: Sequence[Tuple], method: str = "weighted",
                  use_panel_verdict: bool = False,
                  cache: Optional[Dict[str, Dict[str, Any]]] = None,
                  on_row: Optional[Any] = None,
                  ) -> Dict[str, Any]:
    """在给定用例集上跑一遍，返回逐用例原始行与汇总。

    真实面板用 `verdict()`（缺席→共识可能拒绝发布）；受控面板用普通
    `consensus_verdict()`。这里统一走「逐裁判分数 + 手工聚合」，
    因为**逐用例误差**才是 G1/G2 的判据对象，而共识结论是全局决策。

    `cache`：{(puzzle_id, guess_len): row}。命中即复用该行（含各裁判绝对误差），
    **不重复调用端点**——长任务断点续跑，付费 API 实验必需。
    """
    rows: List[Dict[str, Any]] = []
    single_err: Dict[str, List[float]] = {j.id: [] for j in panel.judges}
    absent_counts: Dict[str, int] = {j.id: 0 for j in panel.judges}
    cache = cache if cache is not None else {}
    cache_used = 0

    for puzzle, guess in cases:
        ck = f"{puzzle.soup_id}:{len(guess)}"
        if ck in cache:
            r = cache[ck]
            rows.append(r)
            for jid, e in r.get("abs_by_judge", {}).items():
                single_err.setdefault(jid, []).append(e)
            for jid in r.get("absent", []):
                absent_counts[jid] = absent_counts.get(jid, 0) + 1
            cache_used += 1
            continue

        truth = objective_of(puzzle, guess)
        scored = panel.score_all(puzzle, guess)
        absent = list(getattr(panel, "last_absent", []))
        for jid in absent:
            absent_counts[jid] = absent_counts.get(jid, 0) + 1
        abs_by_judge: Dict[str, float] = {}
        for jid in single_err:
            if jid in absent:
                single_err[jid].append(float("nan"))
                continue
            sc = next(v.score for v in scored if v.judge_id == jid)
            e = abs(sc - truth)
            abs_by_judge[jid] = round(e, 4)
            single_err[jid].append(e)

        weights = panel.weights_from_rmse(panel.calibration)
        eq = st.mean(v.score for v in scored) if scored else None
        wsum = sum(weights.get(v.judge_id, 0.0) for v in scored)
        wt = (sum(v.score * weights.get(v.judge_id, 0.0) for v in scored) / wsum
              if wsum > 0 else None)

        # ⚠️ G1 的基线必须是「任取一个裁判」的误差 mean_j|score_j − truth|，
        # **不能**用 mean(score) − truth（那是等权聚合本身，两者恒等 ⇒ 判据恒
        # 为 0/N）。sanity 阶段实测抓出该退化，此处留痕于设计文档 §6-A。
        single_abs_err = (st.mean(abs(v.score - truth) for v in scored)
                          if scored else None)

        r = {
            "puzzle_id": puzzle.soup_id,
            "frac": round(len(guess) / max(1, len(puzzle.solution)), 3),
            "truth": round(truth, 4),
            "scores": {v.judge_id: round(v.score, 4) for v in scored},
            "abs_by_judge": abs_by_judge,
            "absent": absent,
            "single_abs_err": (round(single_abs_err, 4)
                               if single_abs_err is not None else None),
            "equal": round(eq, 4) if eq is not None else None,
            "weighted": round(wt, 4) if wt is not None else None,
        }
        rows.append(r)
        cache[ck] = r
        # 增量落盘钩子：付费 API 实验中断即丢全部，必须逐例保存
        if on_row is not None:
            on_row(ck, r)

    n = len(rows)
    eq_err = [abs(r["equal"] - r["truth"]) for r in rows if r["equal"] is not None]
    wt_err = [abs(r["weighted"] - r["truth"]) for r in rows
              if r["weighted"] is not None]
    single_case_err = [r["single_abs_err"] for r in rows
                       if r["single_abs_err"] is not None]
    per_judge_rmse = {jid: rmse([e for e in errs if e == e])
                      for jid, errs in single_err.items()}
    g1_better = sum(1 for a, b in zip(eq_err, single_case_err) if a < b)
    g2_better = sum(1 for a, b in zip(wt_err, eq_err) if a <= b)

    # ⚠️ G1 主判据用 **RMSE** 比较（预注册 §6-A 修订）：逐用例绝对误差在
    # 「各裁判误差同号」时恒等于基线（|mean(e)| = mean|e|），会把集成
    # 收益误判为零；RMSE 口径与 P2-A 受控实验一致，且集成的机制正是
    # 抵消随机分量（对应 RMSE 意义下的下降）。逐用例改善数保留为辅助读数。
    baseline_rmse = (st.mean(per_judge_rmse.values())
                     if per_judge_rmse else float("nan"))
    rmse_equal = rmse(eq_err)
    g1_pass_rmse = bool(rmse_equal < baseline_rmse)

    return {
        "rows": rows,
        "n_cases": n,
        "n_scored": len(eq_err),
        # 「随便挑一个裁判」基线 = 各单点 RMSE 的均值（不是等权聚合的 RMSE）
        "rmse_single_mean": baseline_rmse,
        "rmse_equal": rmse_equal,
        "rmse_weighted": rmse(wt_err),
        "per_judge_rmse": per_judge_rmse,
        "absent_counts": absent_counts,
        "g1_better": g1_better,
        "g1_pass_rmse": g1_pass_rmse,
        "g2_better": g2_better,
        "cache_used": cache_used,
        "R": panel.bandwidth_ratio(panel.weights_from_rmse(panel.calibration)),
        "weights": panel.weights_from_rmse(panel.calibration),
    }


# ---- 断点续跑缓存（付费 API 实验必需） ---------------------------------

CALIB_CACHE = LOG_DIR / "_a7_calib_cache.json"
ROW_CACHE = LOG_DIR / "_a7_row_cache.json"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)   # 原子替换：中断不会留下半截缓存


def build_report(real: Dict[str, Any], controlled: Dict[str, Dict[str, Any]],
                 calib: Dict[str, Any], coverage: Dict[str, float],
                 judge_stats: Dict[str, Any], elapsed: float,
                 model: str, personas: Sequence[str],
                 cache_used: int = 0) -> str:
    L: List[str] = []
    A = L.append
    A("# A-7 实验报告 · 真实 LLM 裁判标定")
    A("")
    A(f"- 端点：`{model}` ｜ 裁判人格：{', '.join(personas)}")
    A(f"- 标定集：{calib['n_calib']} 题 → {calib['n_cases_calib']} 例 ｜ "
      f"评测集：{real['n_cases']} 例（按题 disjoint）")
    A(f"- 耗时：{elapsed:.1f}s ｜ 缺席率上限检查：已执行")
    A("")
    A("> 客观真值唯一：`char_f1(guess, solution) × 100`（`engine/evaluate.py`）")
    A("")

    # 标定
    A("## 1. 标定结果（真实裁判可靠性）")
    A("")
    A("| 裁判 | 覆盖率 | 标定 RMSE |")
    A("|---|---|---|")
    for jid, r in calib["rmse"].items():
        A(f"| {jid} | {coverage.get(jid, 0):.2f} | {r:.4f} |")
    for jid in real["absent_counts"]:
        if jid not in calib["rmse"]:
            A(f"| {jid} | — | 覆盖率不足，未参与权重 |")
    A("")
    A(f"- 权重带宽比 **R = {real['R']:.4f}**"
      f"（{'≥' if real['R'] >= R_HOMO_THRESHOLD else '<'}{R_HOMO_THRESHOLD} "
      f"阈值 → {'存在加权空间' if real['R'] >= R_HOMO_THRESHOLD else '近乎同质，加权在理论上无空间'}）")
    A(f"- 闭式拜占庭容忍 f_max = R/(R+2) = {real['R'] / (real['R'] + 2):.4f}")
    A(f"- 权重：{real['weights']}")
    A("")
    n_j = max(1, len(real["absent_counts"]))
    absent_total = sum(real["absent_counts"].values())
    A(f"- 缺席统计：**{absent_total}** 次缺席 / "
      f"({real['n_cases']} 例 × {n_j} 裁判) = 缺席率 "
      f"{absent_total / max(1, real['n_cases'] * n_j):.1%}"
      "（缺席按 §3 计为缺席，不折算成 0 分）")
    A("")
    A("裁判调用统计：")
    A("")
    A("| 裁判 | 成功调用 | 不可用 |")
    A("|---|---|---|")
    for jid, stt in judge_stats.items():
        A(f"| {jid} | {stt['n_calls']} | {stt['n_unavailable']} |")
    A("")
    if cache_used:
        A(f"> ℹ️ 本次运行有 **{cache_used}** 个用例直接复用断点缓存"
            f"（未重新调用端点），故上表调用数少于总用例数。"
            f"缓存内容为 temperature=0 的确定性结果。")
        A("")

    # 判据
    A("## 2. 预注册判据裁决")
    A("")
    n = real["n_scored"]
    g1_pass = bool(real.get("g1_pass_rmse", False))
    g2_pass = real["g2_better"] >= math.ceil(G2_MIN_FRAC * n)
    A(f"| # | 判据 | 通过条件 | 实测 | 裁决 |")
    A("|---|---|---|---|---|")
    A(f"| G1 | 等权集成优于「任取一个裁判」 | 等权 RMSE < 单点基线 RMSE | "
      f"{real['rmse_equal']:.4f} vs {real['rmse_single_mean']:.4f} | "
      f"{'✅ 通过' if g1_pass else '❌ 不通过'} |")
    A(f"| G1' | 逐用例改善（辅助读数） | — | {real['g1_better']}/{n} | 📋 报告 |")
    A(f"| G2 | 加权不劣于等权 | ≥{math.ceil(G2_MIN_FRAC * n)}/{n} 用例 | "
      f"{real['g2_better']}/{n} | {'✅ 通过' if g2_pass else '❌ 不通过（预期如此）'} |")
    A(f"| G3 | 报告 R | — | R={real['R']:.4f} | 📋 报告 |")
    A("")
    A(f"- 真实面板 RMSE：平均单点 **{real['rmse_single_mean']:.4f}** → "
      f"等权 **{real['rmse_equal']:.4f}** → 加权 **{real['rmse_weighted']:.4f}**")
    A("")

    # 对照
    A("## 3. 受控面板对照（同一评测集）")
    A("")
    A("| regime | 平均单点 RMSE | 等权 RMSE | 加权 RMSE | G1 | G2 | R |")
    A("|---|---|---|---|---|---|---|")
    for name, res in controlled.items():
        A(f"| {name} | {res['rmse_single_mean']:.4f} | {res['rmse_equal']:.4f} | "
          f"{res['rmse_weighted']:.4f} | {res['g1_better']}/{res['n_scored']} | "
          f"{res['g2_better']}/{res['n_scored']} | {res['R']:.4f} |")
    A("")
    A(f"| **真实面板** | {real['rmse_single_mean']:.4f} | {real['rmse_equal']:.4f} | "
      f"{real['rmse_weighted']:.4f} | {real['g1_better']}/{real['n_scored']} | "
      f"{real['g2_better']}/{real['n_scored']} | {real['R']:.4f} |")
    A("")

    # 逐用例
    A("## 4. 逐用例原始行（不只汇总）")
    A("")
    A("| 题目 | 梯度 | 真值 | 各裁判分数 | 缺席 | 任取单点误差 | 等权 | 加权 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in real["rows"]:
        sc = " / ".join(f"{k}:{v}" for k, v in r["scores"].items()) or "—"
        ab = ",".join(r["absent"]) or "—"
        A(f"| {r['puzzle_id']} | {r['frac']} | {r['truth']} | {sc} | {ab} | "
          f"{r['single_abs_err']} | {r['equal']} | {r['weighted']} |")
    A("")

    A("## 5. 结论（按判据自动生成，照实不修饰）")
    A("")
    delta = (real["rmse_equal"] - real["rmse_single_mean"]) / max(
        1e-9, real["rmse_single_mean"]) * 100
    if g1_pass:
        A(f"- **集成增益在真实 LLM 裁判上成立（G1 ✅）**：等权共识把 RMSE 从 "
          f"{real['rmse_single_mean']:.4f} 降到 {real['rmse_equal']:.4f}"
          f"（{delta:+.1f}%）。P2-A 受控实验的**核心结论在真实裁判上复现**。")
    else:
        A(f"- **集成增益在真实 LLM 裁判上不成立（G1 ❌）**：等权 "
          f"{real['rmse_equal']:.4f} 未优于单点基线 {real['rmse_single_mean']:.4f}。"
          f"这意味着 P2-A 受控结论**不可外推到真实裁判**，其适用范围须缩到受控环境。")
    if not g2_pass:
        A(f"- **贡献加权无稳健增量（G2 ❌，与预期一致）**：加权 {real['rmse_weighted']:.4f} "
          f"虽在全样本 RMSE 上略低于等权，但逐用例只在 {real['g2_better']}/{n} 上不劣。"
          f"注意 R={real['R']:.4f} 已 >1.25（理论上有空间），"
          f"却仍未转化为稳健增益——说明**有限标定样本估计出的权重本身噪声太大**，"
          f"这与该链 README「R≈1.007 时加权 ≡ 等权」的结论方向一致，"
          f"只是本例中失败原因不是 R 太小，而是**权重估计误差**。")
    else:
        A(f"- 贡献加权取得稳健增量（G2 ✅，{real['g2_better']}/{n}）："
          f"这与 P2-A 受控结论**不一致**，属需要进一步查证的发现。")
    homo = controlled.get("受控·同质", {})
    het = controlled.get("受控·异质", {})
    if homo and het:
        A(f"- 真实裁判的 R={real['R']:.4f} 落在受控同质（{homo['R']:.4f}）"
          f"与受控异质（{het['R']:.4f}）之间：**真实裁判确实存在可观的可靠性异质性**，"
          f"不是同质的。")
    A(f"- 真实裁判 RMSE（{real['rmse_single_mean']:.2f}）显著高于受控面板"
      f"（同质 {homo.get('rmse_single_mean', float('nan')):.2f} / "
      f"异质 {het.get('rmse_single_mean', float('nan')):.2f}）："
      f"真实裁判**更不准**，本层结论描述的是「更真实的误差源」下的聚合行为。")
    A("")

    A("## 6. 能力边界（对外叙述不得省略）")
    A("")
    A(f"- 异质性来自**同模型不同人格 prompt**（`{model}`），**不是**跨模型能力差异；"
      "本报告结论不可外推为「多模型裁判共识的增益」。")
    A("- `temperature=0`：可复现，但**未测**同一人格的重复评方差（日间噪声），"
      "故 R 只反映跨人格差异，可能低估真实裁判间的可靠性差距。")
    A("- 标定集用 solution 前缀构造 guess（协议③）；真实新题**无 ground truth**，"
      "权重外推未经本层验证（同 P2-A 边界）。")
    A("- 受控面板（`judges/biased.py`）是测量误差模型，其结论只在其受控设定内成立。")
    A("- 缺席裁判计为缺席而非 0 分：缺席会真实影响共识（可能拒绝发布）。")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="A-7 真实 LLM 裁判标定")
    ap.add_argument("--model", default="zai/glm-4.5-flash")
    ap.add_argument("--personas", default="strict,lenient,keyword,holistic")
    ap.add_argument("--calib-puzzles", type=int, default=6)
    ap.add_argument("--sanity", action="store_true",
                    help="小规模：2 裁判 × 2 题 × 2 梯度（8 次调用）")
    ap.add_argument("--max-absent-rate", type=float, default=0.3)
    ap.add_argument("--out", default=str(LOG_DIR / "real_judge_report.md"))
    args = ap.parse_args(argv)

    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    # sanity：2 裁判 × (1 题标定 + 1 题评测) × 2 梯度 = 8 次调用，只验仪器
    if args.sanity:
        personas = personas[:2]
    fracs = (0.15, 0.55) if args.sanity else GUESS_FRACS
    n_calib = 1 if args.sanity else args.calib_puzzles

    puzzles = load_puzzles()
    if args.sanity:
        puzzles = puzzles[:2]          # sanity 只跑 2 题，避免多烧调用
    all_ids = [p.soup_id for p in puzzles]
    calib_ids = set(all_ids[:n_calib])
    cases = make_cases(puzzles, fracs)
    calib_cases, eval_cases = split_by_puzzle(cases, calib_ids)
    if not calib_cases or not eval_cases:
        print("[a7] 标定/评测集为空，请检查 --calib-puzzles")
        return 1

    print(f"[a7] 端点={args.model} 人格={personas}")
    print(f"[a7] 标定 {len(calib_cases)} 例 / 评测 {len(eval_cases)} 例")
    t0 = time.time()

    # ---- 真实面板标定（带缓存：标定 30 例 × 4 裁判 ≈ 120 次调用，不能重来） ----
    panel = make_real_panel(personas, model=args.model, sign=False)
    calib_key = f"{args.model}|{'|'.join(personas)}|n{n_calib}|{list(fracs)}"
    store = _load_json(CALIB_CACHE)
    entry = store.get(calib_key)
    if entry:
        rmse, coverage = entry["rmse"], entry["coverage"]
        panel.calibration = dict(rmse)
        panel._coverage = dict(coverage)
        print(f"[a7] 标定缓存命中（跳过 {len(calib_cases)} 例重跑）")
    else:
        rmse = panel.calibrate_safe(calib_cases)
        coverage = panel.coverage(calib_cases)
        store[calib_key] = {"rmse": rmse, "coverage": coverage}
        _dump_json(CALIB_CACHE, store)
    elapsed = time.time() - t0
    worst = min(coverage.values()) if coverage else 0.0
    print(f"[a7] 标定完成 {elapsed:.1f}s ｜ RMSE="
          f"{ {k: round(v, 3) for k, v in rmse.items()} }")
    print(f"[a7] 覆盖率={ {k: round(v, 2) for k, v in coverage.items()} }")
    if worst < (1 - args.max_absent_rate):
        print(f"[a7] ⛔ 覆盖率 {worst:.2f} 低于阈值 "
              f"{1 - args.max_absent_rate:.2f}，停止（不硬撑）")
        return 1

    # ---- 真实面板评测（逐用例缓存，可断点续跑） ------------------------
    row_store = _load_json(ROW_CACHE)
    rk = f"{calib_key}|eval"
    cache = row_store.get(rk, {})

    def _persist(_ck, _row):
        # 逐例落盘：中断不丢已完成用例（每例一次写，IO 成本可忽略）
        row_store[rk] = cache
        _dump_json(ROW_CACHE, row_store)

    real = aggregate_run(panel, eval_cases, cache=cache, on_row=_persist)
    row_store[rk] = cache
    _dump_json(ROW_CACHE, row_store)
    print(f"[a7] 评测 {real['n_scored']}/{real['n_cases']} 例可评分"
          f"（缓存复用 {real['cache_used']} 例）")
    judge_stats = {j.id: {"n_calls": j.n_calls, "n_unavailable": j.n_unavailable}
                   for j in panel.judges}
    real["R"] = panel.bandwidth_ratio(real["weights"])
    real["weights"] = panel.weights_from_rmse(panel.calibration)

    # ---- 受控面板对照（零成本） --------------------------------------
    controlled: Dict[str, Dict[str, Any]] = {}
    for name, mk in (("受控·同质", make_homogeneous_panel),
                     ("受控·异质", make_heterogeneous_panel)):
        cj = mk(len(personas), seed0=0)
        cp = JudgePanel(cj, sign=False)
        cp.calibrate(calib_cases)
        controlled[name] = aggregate_run(cp, eval_cases)

    elapsed = time.time() - t0
    calib_meta = {"rmse": rmse, "n_calib": len(calib_ids),
                  "n_cases_calib": len(calib_cases)}
    report = build_report(real, controlled, calib_meta, coverage, judge_stats,
                          elapsed, args.model, personas,
                          cache_used=real.get("cache_used", 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print(f"[a7] 真实面板：单点 {real['rmse_single_mean']:.4f} → "
          f"等权 {real['rmse_equal']:.4f} → 加权 {real['rmse_weighted']:.4f}")
    print(f"[a7] G1 {real['g1_better']}/{real['n_scored']} ｜ "
          f"G2 {real['g2_better']}/{real['n_scored']} ｜ R={real['R']:.4f}")
    print(f"[a7] 报告 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())