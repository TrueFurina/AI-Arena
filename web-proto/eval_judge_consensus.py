"""多裁判共识评测（P2-A）——回答「共识裁决是否比单点裁判更可信」。

## 要回答的问题（可证伪，不是演示）

  Q1 加权共识裁决 vs 单点裁判 vs 等权平均，谁更贴近客观真值？
  Q2 增益是否依赖裁判可靠性的**异质性**？（同质 ⇒ 预期无增益）

Q2 是诚实性的关键。真实链 README 自述：R≈1.007 时 CW-PBFT ≡ 标准 PBFT。
若本层在**同质**裁判上也报出增益，那多半是标定噪声被当成了信号——
所以同质组是**零效应对照组**，预期增益 ≈ 0，不是失败。

## 三条协议（违反任一条结论即无效）

  ① **标定集与评测集按题 disjoint**：用被测数据标定自己 = 循环论证
  ② **客观真值唯一**：char_f1(guess, solution)×100，与 engine/evaluate.py 同源
  ③ **guess 不得取完整 solution**：那会让真值 = 100，而分数被 [0,100] 截断后
     噪声只剩半边 → RMSE 被系统性低估，且同质裁判也被估出差异（R 假性 > 1）。
     实测踩过：取整段 solution 时 4 个 σ=8 的同质裁判 RMSE 竟为 6.07/4.36/
     5.07/3.27（全部低于 8），R = 1.18 而非 1.0。故用 solution 前缀构造梯度。

## 输出

  logs/judge_consensus_report.md —— 含样本阶梯（标定集大小）与多 seed 结果
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from engine.evaluate import char_f1
from engine.puzzles import load_puzzles
from judges.biased import (JudgeProfile, BiasedJudge,
                           make_homogeneous_panel, make_heterogeneous_panel)
from judges.panel import JudgePanel

LOG_DIR = Path(__file__).parent / "logs"

# 客观真值：唯一口径，禁止在本脚本另算
def objective_of(puzzle, guess: str) -> float:
    return char_f1(guess, puzzle.solution) * 100.0


# --------------------------------------------------------------------------
# 数据构造
# --------------------------------------------------------------------------

# 刻意不取 1.0：见模块文档协议 ③
GUESS_FRACS = (0.15, 0.35, 0.55, 0.75, 0.90)


def make_cases(puzzles: Sequence) -> List[Tuple]:
    """构造 (puzzle, guess) 用例：solution 前缀形成质量梯度。"""
    cases = []
    for p in puzzles:
        n = len(p.solution)
        for f in GUESS_FRACS:
            cases.append((p, p.solution[:max(1, int(n * f))]))
    return cases


def split_by_puzzle(cases: Sequence[Tuple], train_ids: set):
    """按题目划分，保证标定/评测无泄漏（协议 ①）。"""
    tr = [c for c in cases if c[0].soup_id in train_ids]
    te = [c for c in cases if c[0].soup_id not in train_ids]
    return tr, te


# --------------------------------------------------------------------------
# 评测
# --------------------------------------------------------------------------

def rmse_aggregate(panel: JudgePanel, weights: Dict[str, float],
                   cases: Sequence[Tuple], method: str) -> float:
    """按给定聚合方式计算裁决相对客观真值的 RMSE。"""
    errs = []
    for puzzle, guess in cases:
        truth = objective_of(puzzle, guess)
        vs = panel.score_all(puzzle, guess)
        if method == "equal":
            agg = sum(v.score for v in vs) / len(vs)
        else:
            wsum = sum(weights[v.judge_id] for v in vs)
            agg = sum(v.score * weights[v.judge_id] for v in vs) / wsum
        errs.append((agg - truth) ** 2)
    return math.sqrt(sum(errs) / len(errs))


def rmse_single(panel: JudgePanel, cases: Sequence[Tuple]) -> Dict[str, float]:
    """各单点裁判的 RMSE（用于「随便挑一个裁判」基线）。"""
    e2 = {j.id: [] for j in panel.judges}
    for puzzle, guess in cases:
        truth = objective_of(puzzle, guess)
        for j in panel.judges:
            sc, _ = j.score(puzzle, guess)
            e2[j.id].append((sc - truth) ** 2)
    return {k: math.sqrt(sum(v) / len(v)) for k, v in e2.items()}


def run_regime(name: str, mk_panel, n_judges: int, puzzles, seeds,
               n_calib_puzzles: int, repeats: int = 3):
    """跑一个 regime（同质/异质），跨 seed 汇总。"""
    all_ids = [p.soup_id for p in puzzles]
    calib_ids = set(all_ids[:n_calib_puzzles])
    eval_ids = set(all_ids[n_calib_puzzles:])
    cases = make_cases(puzzles)
    calib_cases, eval_cases = split_by_puzzle(cases, calib_ids)

    rows = []
    for seed in seeds:
        judges = mk_panel(n_judges, seed0=seed * 100)
        panel = JudgePanel(judges, sign=False)  # 标定循环不开签名（省时）

        single = rmse_single(panel, calib_cases)
        weights = JudgePanel.weights_from_rmse(single)
        R = JudgePanel.bandwidth_ratio(weights)

        # 评测集上（与标定集按题 disjoint）
        e_single = rmse_single(panel, eval_cases)
        e_equal = rmse_aggregate(panel, weights, eval_cases, "equal")
        e_weighted = rmse_aggregate(panel, weights, eval_cases, "weighted")

        rows.append({
            "seed": seed,
            "R": R,
            "f_max": JudgePanel.fault_tolerance(R),
            "single_mean": st.mean(e_single.values()),
            "single_best": min(e_single.values()),
            "equal": e_equal,
            "weighted": e_weighted,
            "gain_equal_minus_weighted": e_equal - e_weighted,
            "gain_single_minus_weighted": st.mean(e_single.values()) - e_weighted,
        })
    return rows


def run_calibration_ladder(mk_panel, n_judges: int, puzzles,
                           sizes: Sequence[int], seed: int = 0):
    """标定集样本阶梯：R 应随标定样本增多而收敛到真实异质性。"""
    all_ids = [p.soup_id for p in puzzles]
    cases = make_cases(puzzles)
    out = []
    for n in sizes:
        calib_ids = set(all_ids[:n])
        eval_ids = set(all_ids[n:])
        if not eval_ids:
            continue
        calib_cases, _ = split_by_puzzle(cases, calib_ids)
        judges = mk_panel(n_judges, seed0=seed)
        panel = JudgePanel(judges, sign=False)
        rmse = rmse_single(panel, calib_cases)
        w = JudgePanel.weights_from_rmse(rmse)
        out.append({
            "n_calib_puzzles": n,
            "n_calib_cases": len(calib_cases),
            "R": JudgePanel.bandwidth_ratio(w),
            "rmse": {k: round(v, 3) for k, v in rmse.items()},
        })
    return out


# --------------------------------------------------------------------------
# 共识拒绝行为（L3 内核属性）
# --------------------------------------------------------------------------

def run_refusal_demo(puzzles, n_judges: int = 4, seed: int = 0):
    """展示「权重不足即拒绝发布」——移除链就没有这个行为。

    对比两种缺席情形：
      A 缺席的是低可靠裁判 → 剩余权重充足 → 仍发布
      B 缺席的是高可靠裁判 → 剩余权重不足 → 拒绝发布
    单点裁判不存在这种取舍：它永远返回一个数。
    """
    judges = make_heterogeneous_panel(n_judges, seed0=seed)
    panel = JudgePanel(judges, sign=False)
    calib = panel.calibrate([(p, p.solution[:max(1, int(len(p.solution) * 0.5))])
                             for p in puzzles[:8]])
    w = panel.weights_from_rmse(calib)
    ranked = sorted(w.items(), key=lambda kv: -kv[1])

    p = puzzles[-1]
    guess = p.solution[:max(1, int(len(p.solution) * 0.5))]
    out = []
    # 全在场
    v = panel.consensus_verdict(p, guess)
    out.append({"case": "全在场", "absent": [], "accepted": v.accepted,
                "score": v.score, "ratio": v.detail.get("responded_ratio")})
    # 缺席一个最低可靠
    worst = [ranked[-1][0]]
    v = panel.consensus_verdict(p, guess, absent=worst)
    out.append({"case": f"缺席最低可靠({worst[0]})", "absent": worst,
                "accepted": v.accepted, "score": v.score,
                "ratio": v.detail.get("responded_ratio")})
    # 缺席两个最高可靠
    top2 = [kv[0] for kv in ranked[:2]]
    v = panel.consensus_verdict(p, guess, absent=top2)
    out.append({"case": f"缺席最高可靠({','.join(top2)})", "absent": top2,
                "accepted": v.accepted, "score": v.score,
                "ratio": v.detail.get("responded_ratio")})
    return out, w


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-judges", type=int, default=4)
    ap.add_argument("--n-calib-puzzles", type=int, default=6)
    ap.add_argument("--ladder", type=int, nargs="*",
                    default=[2, 4, 6, 8, 10])
    args = ap.parse_args()

    puzzles = load_puzzles()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[judge-consensus] 题库 {len(puzzles)} 题 | 裁判 {args.n_judges} 名 "
          f"| 标定题 {args.n_calib_puzzles} | seeds={args.seeds}")
    print()

    regimes = {
        "同质(sigma=8 全部相同)": make_homogeneous_panel,
        "异质(sigma=2/8/20/35 + 一人自偏好)": make_heterogeneous_panel,
    }
    results = {}
    for name, mk in regimes.items():
        rows = run_regime(name, mk, args.n_judges, puzzles, args.seeds,
                          args.n_calib_puzzles)
        results[name] = rows
        print(f"=== {name} ===")
        print(" seed |     R  | f_max | 单点(均) | 单点(最优) |  等权  |  加权  "
              "| 等权-加权 | 单点-加权")
        for r in rows:
            print(f"  {r['seed']}   | {r['R']:.3f} | {r['f_max']:.3f} | "
                  f"{r['single_mean']:7.3f} | {r['single_best']:9.3f} | "
                  f"{r['equal']:6.3f} | {r['weighted']:6.3f} | "
                  f"{r['gain_equal_minus_weighted']:+9.3f} | "
                  f"{r['gain_single_minus_weighted']:+9.3f}")
        g = [r["gain_equal_minus_weighted"] for r in rows]
        gs = [r["gain_single_minus_weighted"] for r in rows]
        print(f"  加权 vs 等权 增益: 均值 {st.mean(g):+.4f} "
              f"[{min(g):+.4f}, {max(g):+.4f}]  全为正={all(x > 0 for x in g)}")
        print(f"  加权 vs 单点 增益: 均值 {st.mean(gs):+.4f} "
              f"[{min(gs):+.4f}, {max(gs):+.4f}]  全为正={all(x > 0 for x in gs)}")
        # 增益分解：**本次最要紧的一条**
        ens = [r["single_mean"] - r["equal"] for r in rows]
        wt = g
        tot = [a + b for a, b in zip(ens, wt)]
        share = [b / t for b, t in zip(wt, tot) if t > 0]
        if share:
            print(f"  【增益分解】集成(单点→等权) {st.mean(ens):+.4f} | "
                  f"加权(等权→加权) {st.mean(wt):+.4f} | "
                  f"加权占比 {st.mean(share) * 100:.1f}%")
            print("             → 收益主体来自**多裁判集成**，不是贡献加权")
        print()

    print("=== 标定集样本阶梯（R 应随标定样本增多收敛）===")
    ladder = {}
    for name, mk in regimes.items():
        lad = run_calibration_ladder(mk, args.n_judges, puzzles, args.ladder)
        ladder[name] = lad
        print(f"  {name}")
        for L in lad:
            print(f"    标定 {L['n_calib_puzzles']} 题/{L['n_calib_cases']} 用例 "
                  f"→ R={L['R']:.4f}  RMSE={L['rmse']}")
    print()

    print("=== 共识拒绝行为（移除链即失去该行为）===")
    refusal, w = run_refusal_demo(puzzles, args.n_judges)
    print(f"  权重: {{{', '.join(f'{k}:{v:.3f}' for k, v in w.items())}}}")
    for r in refusal:
        print(f"  {r['case']:<28} 响应权重比={r['ratio']} "
              f"accepted={r['accepted']} score={r['score']}")
    print()

    # 落盘报告
    lines = ["# 多裁判共识评测（P2-A）\n",
             f"- 题库 {len(puzzles)} 题，裁判 {args.n_judges} 名，"
             f"标定题 {args.n_calib_puzzles}，seeds={args.seeds}\n",
             "- 客观真值口径：`char_f1(guess, solution)×100`\n",
             "- 标定集与评测集**按题 disjoint**\n\n",
             "## 主结果\n\n",
             "| regime | R | f_max | 单点(均) | 单点(最优) | 等权 | 加权 | "
             "等权−加权 | 单点−加权 |\n|---|---|---|---|---|---|---|---|---|\n"]
    for name, rows in results.items():
        for r in rows:
            lines.append(f"| {name} (seed{r['seed']}) | {r['R']:.3f} | "
                         f"{r['f_max']:.3f} | {r['single_mean']:.3f} | "
                         f"{r['single_best']:.3f} | {r['equal']:.3f} | "
                         f"{r['weighted']:.3f} | "
                         f"{r['gain_equal_minus_weighted']:+.3f} | "
                         f"{r['gain_single_minus_weighted']:+.3f} |\n")
    lines.append("\n## 标定集样本阶梯\n\n")
    for name, lad in ladder.items():
        lines.append(f"**{name}**\n\n| 标定题 | 用例数 | R |\n|---|---|---|\n")
        for L in lad:
            lines.append(f"| {L['n_calib_puzzles']} | {L['n_calib_cases']} | "
                         f"{L['R']:.4f} |\n")
        lines.append("\n")
    lines.append("\n## 共识拒绝行为\n\n| 情形 | 响应权重比 | accepted | score |\n|---|---|---|---|\n")
    for r in refusal:
        lines.append(f"| {r['case']} | {r['ratio']} | {r['accepted']} | "
                     f"{r['score']} |\n")
    (LOG_DIR / "judge_consensus_report.md").write_text(
        "".join(lines), encoding="utf-8")
    print(f"[judge-consensus] 报告已写入 {LOG_DIR / 'judge_consensus_report.md'}")


if __name__ == "__main__":
    main()
