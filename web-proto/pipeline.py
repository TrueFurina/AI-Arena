"""P3 · 一键流水线：跑评测 → 出指标 → 多裁判共识 → 上链凭证 → 出报告。

用法：
    python pipeline.py                                   # 零 key / 零链默认可跑
    python pipeline.py --panel hetero --identity          # 共识面板 + 身份锚定
    python pipeline.py --puzzle turtle_soup_003 --panel hetero --absent j0,j1
    python pipeline.py --no-chain                         # 强制仅本地签名

退出码：0 = 完成（允许降级）；1 = 凭证验签失败或构造错误（不可降级）。
设计文档：`DESIGN-P3-PIPELINE.md`（三条设计判断 + 六条预注册验收 F1–F6）。

## 三条不可让渡的设计约束（改动前先读设计文档）

1. 共识 `accepted=False` 时**不得**用任何分数填补（判断 1）——
   「拒绝发布」是共识层相对单点裁判的唯一实质价值
2. 共识结论写进**被签名的** `verdict`（判断 2）——否则改报告就能改数字
3. `tx_id` 只是链上索引，**不能凭空验签**（判断 3）——那是前端的事，
   但本流水线在报告里必须给出可离线验签的入口
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from engine.evaluate import metrics_from_result
from engine.game import TurtleSoupGame
from engine.puzzles import get_puzzle, load_puzzles
from players import make_player

from chain.recorder import MatchRecorder
from chain.verify import verify_credential
from chain.identity import IdentityRegistry
from chain.models import Credential

from judges.biased import make_homogeneous_panel, make_heterogeneous_panel
from judges.panel import ConsensusVerdict, JudgePanel

from pipeline_report import render_report

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"
CRED_DIR = LOG_DIR / "credentials"

# 标定集梯度（沿用 eval_judge_consensus 协议③：刻意不取 1.0，
# 取整段 solution 会让真值=100 被 [0,100] 截断，噪声只剩半边 → RMSE 被低估）
GUESS_FRACS = (0.15, 0.35, 0.55, 0.75, 0.90)


@dataclass
class PipelineResult:
    game_id: str
    game: Dict[str, Any]
    metrics: Dict[str, Any]
    consensus: Optional[ConsensusVerdict] = None
    credential: Optional[Credential] = None
    verification: Optional[Any] = None
    game_path: Optional[Path] = None
    credential_path: Optional[Path] = None
    report_path: Optional[Path] = None
    degraded: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """凭证缺失不算失败（可能显式关闭）；有凭证则必须验签通过。"""
        if self.credential is None:
            return True
        return bool(self.verification is not None and self.verification.ok)


def _calibration_cases(puzzles: Sequence, exclude_id: str, n_puzzles: int):
    """构造标定用例，保证与被测题 disjoint（协议①）。"""
    others = [p for p in puzzles if p.soup_id != exclude_id][:n_puzzles]
    cases = []
    for p in others:
        n = len(p.solution)
        for f in GUESS_FRACS:
            cases.append((p, p.solution[: max(1, int(n * f))]))
    return cases


def _final_guess(game: Dict[str, Any]) -> str:
    for m in game.get("transcript", []):
        if m.get("role") == "guess":
            return str(m.get("content", ""))
    return ""


def run_pipeline(
    puzzle_id: str = "turtle_soup_001",
    questioners: Sequence[str] = ("mock-gpt4o",),
    host_spec: str = "mock-host",
    judge_spec: str = "mock-judge",
    rounds: int = 8,
    panel: Optional[str] = None,
    n_judges: int = 4,
    calib_puzzles: int = 6,
    seed: int = 0,
    absent: Sequence[str] = (),
    submit_to_chain: bool = True,
    use_identity: bool = False,
    chain_path: Optional[str] = None,
    out_dir: Path = LOG_DIR,
    puzzles_path: Optional[str] = None,
) -> PipelineResult:
    """跑完整链路（S1 对局 → S2 指标 → S3 共识 → S4 凭证 → S5 报告）。"""
    degraded: List[str] = []
    out_dir = Path(out_dir)
    (out_dir / "credentials").mkdir(parents=True, exist_ok=True)

    # -- S1 对局 ---------------------------------------------------------
    puzzle = get_puzzle(puzzle_id, path=puzzles_path) if puzzles_path else get_puzzle(puzzle_id)
    host = make_player(host_spec, role="host")
    qs = [make_player(spec, role="questioner", index=i)
          for i, spec in enumerate(questioners)]
    judge = make_player(judge_spec, role="judge") if judge_spec else None

    game = TurtleSoupGame(puzzle, host, qs, judge=judge, max_rounds=rounds).run()
    res = game.to_dict()
    game_path = out_dir / f"pipeline_game_{res['game_id']}.json"
    game_path.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    # -- S2 指标（唯一口径） ---------------------------------------------
    metrics = metrics_from_result(res)

    # -- S3 共识（可选） -------------------------------------------------
    consensus: Optional[ConsensusVerdict] = None
    verdict_extra: Optional[Dict[str, Any]] = None

    if panel:
        mk = {"homo": make_homogeneous_panel,
              "hetero": make_heterogeneous_panel}.get(panel)
        if mk is None:
            raise ValueError(f"未知面板类型：{panel}（可选 none/homo/hetero）")
        judges = mk(n_judges, seed0=seed * 100)
        jpanel = JudgePanel(judges, sign=True, chain_path=chain_path)

        all_puzzles = load_puzzles(puzzles_path) if puzzles_path else load_puzzles()
        calib = _calibration_cases(all_puzzles, puzzle.soup_id, calib_puzzles)
        if not calib:
            raise ValueError("标定集为空：无法标定即不得跑共识（协议①）")
        jpanel.calibrate(calib)

        guess = _final_guess(res)
        consensus = jpanel.consensus_verdict(puzzle, guess, absent=absent)
        if consensus.backend == "local-weighted":
            degraded.append(
                "真实 CW-PBFT 引擎不可用，共识退化为本地 2/3 权重门限"
                "（判据一致，但不具链上记录）")
        if not consensus.accepted:
            degraded.append(
                "共识拒绝发布结论（可靠权重不足/裁判缺席）——按设计判断 1，"
                "不以任何单点或等权分数替代")

        # 判断 2：共识结论进签名对象
        verdict_extra = {
            "mode": "panel-consensus",
            "consensus": consensus.to_dict(),
        }

    # -- S4 凭证 ---------------------------------------------------------
    rec = MatchRecorder(chain_path=chain_path, submit_to_chain=submit_to_chain)
    if rec._chain is None and submit_to_chain:
        degraded.append("真实共识链不可用：凭证仅本地签名，anchor=None（不阻塞）")

    # 裁判标识：Judge 契约只保证 .score，id 并非所有实现都有（如 MockJudge
    # 只有 .model）——取不到就退回规格串，不伪造一个「看起来正规」的 id
    judge_id = "none"
    if judge is not None:
        judge_id = (getattr(judge, "id", None)
                    or getattr(judge, "model", None)
                    or judge_spec)
    cred = rec.issue_from_result(res, judge_id=str(judge_id),
                                 verdict_extra=verdict_extra)

    identity = None
    if use_identity:
        reg = IdentityRegistry(chain_path=chain_path)
        # `available` 是 property（不是方法）：写成 reg.available() 会直接抛
        # TypeError。此处兼容两种形态，避免这类「属性 vs 方法」误用再犯
        avail = reg.available
        if callable(avail):
            avail = avail()
        if avail:
            reg.register_signer(rec.signer)
            identity = reg
        else:
            degraded.append("身份合约不可用：未做签发者身份校验（仅验证完整性）")

    verification = verify_credential(cred, chain_path=chain_path,
                                     identity=identity)
    if identity is not None and verification.identity_status is None:
        # 请求了身份校验却没执行 = 静默跳过，属必须暴露的失败
        degraded.append("已请求身份校验但未产出 identity_status（应检查接线）")

    cred_path = out_dir / "credentials" / f"pipeline_{res['game_id']}.credential.json"
    rec.save(cred, cred_path)

    # -- S5 报告 ---------------------------------------------------------
    artifacts: Dict[str, Any] = {
        "game": res,
        "metrics": metrics,
        "consensus": consensus.to_dict() if consensus else None,
        "credential": cred.to_dict(),
        "verify": {
            "ok": verification.ok,
            "steps": verification.steps,
            "reasons": verification.reasons,
            "identity_status": verification.identity_status,
        },
        "game_path": str(game_path),
        "credential_path": str(cred_path),
        "degraded": degraded,
    }
    report_path = out_dir / f"pipeline_report_{res['game_id']}.md"
    report_path.write_text(render_report(artifacts), encoding="utf-8")

    return PipelineResult(
        game_id=res["game_id"], game=res, metrics=metrics,
        consensus=consensus, credential=cred, verification=verification,
        game_path=game_path, credential_path=cred_path,
        report_path=report_path, degraded=degraded,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AI-Arena 一键评测流水线（P3）")
    ap.add_argument("--puzzle", default="turtle_soup_001")
    ap.add_argument("--questioner", action="append", default=None,
                    help="提问者模型（可重复；mock- 开头为离线内置）")
    ap.add_argument("--host", default="mock-host")
    ap.add_argument("--judge", default="mock-judge",
                    help="引擎内单点裁判（仅参考）；空字符串表示不启用")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--panel", default="none",
                    choices=["none", "homo", "hetero"],
                    help="多裁判共识面板类型")
    ap.add_argument("--n-judges", type=int, default=4)
    ap.add_argument("--calib-puzzles", type=int, default=6,
                    help="标定集题目数（与被测题 disjoint）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--absent", default="",
                    help="缺席裁判 id，逗号分隔（用于演示共识拒绝发布）")
    ap.add_argument("--no-chain", action="store_true", help="不提交共识链")
    ap.add_argument("--identity", action="store_true",
                    help="接入身份锚定（把签发者公钥登记到身份合约）")
    ap.add_argument("--chain-path", default=None)
    ap.add_argument("--out", default=str(LOG_DIR))
    args = ap.parse_args(argv)

    try:
        pr = run_pipeline(
            puzzle_id=args.puzzle,
            questioners=tuple(args.questioner or ["mock-gpt4o"]),
            host_spec=args.host, judge_spec=args.judge,
            rounds=args.rounds,
            panel=(None if args.panel == "none" else args.panel),
            n_judges=args.n_judges, calib_puzzles=args.calib_puzzles,
            seed=args.seed,
            absent=tuple(x.strip() for x in args.absent.split(",") if x.strip()),
            submit_to_chain=not args.no_chain,
            use_identity=args.identity,
            chain_path=args.chain_path, out_dir=Path(args.out),
        )
    except Exception as exc:
        print(f"[pipeline] ❌ 失败：{type(exc).__name__}: {exc}")
        return 1

    print(f"[pipeline] 对局 {pr.game_id} ｜ f1={pr.metrics.get('f1'):.4f} "
          f"｜ dhit={pr.metrics.get('dhit'):.4f}")
    if pr.consensus is not None:
        c = pr.consensus
        state = "已发布" if c.accepted else "拒绝发布"
        print(f"[pipeline] 共识 {state}：score={c.score} ｜ "
              f"等权={c.equal_weight_score} ｜ R={c.R:.3f} ｜ "
              f"后端={c.backend} ｜ 缺席={c.absent or '无'}")
    if pr.credential is not None:
        anchor = pr.credential.anchor or {}
        print(f"[pipeline] 凭证 {str(pr.credential.record_hash)[:16]} ｜ "
              f"锚定={anchor.get('status', '未上链(仅本地签名)')} ｜ "
              f"tx_id={anchor.get('tx_id', '—')}")
        print(f"[pipeline] 验签：{pr.verification.summary()}")
    for d in pr.degraded:
        print(f"[pipeline] ⚠️ {d}")
    print(f"[pipeline] 报告 → {pr.report_path}")

    return 0 if pr.ok else 1


if __name__ == "__main__":
    sys.exit(main())
