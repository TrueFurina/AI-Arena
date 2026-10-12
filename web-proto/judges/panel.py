"""裁判面板：多裁判独立签名 + 可靠性标定 + CW-PBFT 共识（P2-A 核心）。

## 它解决什么

续5 实测：LLM 裁判自评 score 与客观 f1 系统性不一致（自偏好）。
⇒ 单看任何一个裁判的数字都不可信，而 P1 只能保证「这个数字没被改过」。

本层的答案：**结论不再由单个裁判给出，而由面板共识给出**，并且
**在可靠权重不足时拒绝给出**（单点裁判永远会给数字，这是本质差别）。

## 数据流

   裁判独立评分 → 各自 ECDSA 签名 → 可靠性标定（带 ground truth 的题集）
        → 标定权重 → 加权聚合 → 对聚合结果跑 CW-PBFT 共识
        → 共识通过才发布结论，否则标记 rejected（不产出分数）

## 三条不能破的规矩

  1. **标定必须 held-out**：用被测数据标定自己 = 循环论证。calibrate() 与
     evaluate() 的数据集必须分离，由调用方保证（eval_judge_consensus.py
     用 split 明确分开，测试亦有守护）。
  2. **权重公式沿用链自身的 settled-weight 口径**：w = 1.0 + 0.5 × norm，
     ∈ (1.0, 1.5]。不自创公式，否则与那条链的既有结论不可比。
  3. **共识失败不得静默**：rejected 必须写进结论对象，绝不返回「看起来
     正常」的分数。链不可用 → 退化为本地加权聚合，并如实标注 backend。

## 权重与 R 的关系（口径来自该链 README）

  R = w_max / w_min，闭式判据 f_max = R / (R + 2)（拜占庭容忍上限）
  裁判同质 ⇒ R = 1.0 ⇒ f_max = 1/3 ⇒ 与标准 PBFT 无差别（该链已实测）
  本层**不预设增量存在**：同质感光下应观察到增益 ≈ 0，这是预期结果。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from chain.crypto_adapter import (
    DEFAULT_CHAIN_PATH,
    Signer,
    verify_signature,
)
from chain.models import canonical_bytes, sha256_hex
from engine.evaluate import char_f1
from judges.biased import BiasedJudge

# 权重公式常量：沿用真实链的 settled-weight 口径（WEIGHT_BASE / WEIGHT_GAIN）
WEIGHT_BASE = 1.0
WEIGHT_GAIN = 0.5

# RMSE 趋零时的保护项，防除零
_EPS = 1e-9


# --------------------------------------------------------------------------
# 真实共识引擎的软导入
# --------------------------------------------------------------------------

def try_import_pbft(chain_path: Optional[str] = None):
    """尝试导入真实 CW-PBFT 引擎；不可用返回 None（调用方降级）。"""
    p = chain_path or DEFAULT_CHAIN_PATH
    if not p:
        return None
    try:
        import sys
        from pathlib import Path
        if not Path(p).exists():
            return None
        if p not in sys.path:
            sys.path.insert(0, p)
        from blockchain.consensus.cw_pbft import CWPBFTConsensus
        return {"CWPBFTConsensus": CWPBFTConsensus, "path": p}
    except Exception:
        return None


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------

@dataclass
class SignedVerdict:
    """一个裁判的独立裁决（自带签名与公钥，可离线验签）。"""

    judge_id: str
    score: float
    rationale: str
    signature_hex: str
    public_key_hex: str
    algorithm: str
    key_custody: str

    def payload(self) -> Dict[str, Any]:
        # 只签这两个字段：分数与裁判身份。rationale 不签（说明性文本，
        # 改动它不影响结论，但签进去会让人误以为说明也经过认证）
        return {"judge_id": self.judge_id, "score": self.score}

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.payload())

    def verify(self, chain_path: Optional[str] = None) -> bool:
        return verify_signature(self.public_key_hex, self.canonical_bytes(),
                                self.signature_hex, chain_path=chain_path)


@dataclass
class ConsensusVerdict:
    """面板共识结论。

    `accepted=False` 时 `score` 必为 None——**拒绝发布就是拒绝发布**，
    不给一个「凑合能用」的数字。
    """

    # 共识分数；未通过共识时为 None
    score: Optional[float]
    accepted: bool
    # 参与评分的裁判签名裁决
    verdicts: List[SignedVerdict]
    # {judge_id: weight}
    weights: Dict[str, float]
    # 权重带宽比 R = w_max / w_min
    R: float
    # 闭式拜占庭容忍上限 f_max = R / (R + 2)
    f_max: float
    # 未响应（拜占庭/缺席）的裁判
    absent: List[str]
    # 后端：cw-pbft / local-weighted
    backend: str
    # 聚合方式，便于与等权对照
    method: str = "weighted"
    # 等权平均（始终计算，用于对照；不因加权而丢弃）
    equal_weight_score: Optional[float] = None
    # 共识过程诊断
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------
# 面板
# --------------------------------------------------------------------------

class JudgePanel:
    """多裁判面板。

    Parameters
    ----------
    judges       : 裁判列表（需有 .id / .profile.judge_id 与 .score）
    sign         : 是否对每个裁决做 ECDSA 签名（关掉可加速标定循环）
    chain_path   : 真实共识链路径；None 表示用默认路径尝试
    """

    def __init__(self, judges: Sequence[Any], sign: bool = True,
                 chain_path: Optional[str] = None,
                 signer_factory: Optional[Callable[[str], Signer]] = None):
        if not judges:
            raise ValueError("裁判面板不能为空")
        self.judges = list(judges)
        self.sign = sign
        self.chain_path = chain_path
        self._signers: Dict[str, Signer] = {}
        if sign:
            factory = signer_factory or (lambda jid: Signer(signer_id=jid))
            for j in self.judges:
                self._signers[j.id] = factory(j.id)
        self._pbft = try_import_pbft(chain_path)
        self.calibration: Optional[Dict[str, float]] = None

    # -- 标识 -----------------------------------------------------------
    @property
    def judge_ids(self) -> List[str]:
        return [j.id for j in self.judges]

    @property
    def consensus_available(self) -> bool:
        return self._pbft is not None

    # -- 评分 -----------------------------------------------------------
    def score_all(self, puzzle, guess: str,
                  absent: Iterable[str] = ()) -> List[SignedVerdict]:
        """收集所有（未缺席）裁判的独立裁决，逐个签名。"""
        absent_set = set(absent)
        out: List[SignedVerdict] = []
        for j in self.judges:
            if j.id in absent_set:
                continue
            sc, rationale = j.score(puzzle, guess)
            if self.sign:
                s = self._signers[j.id]
                sig = s.sign_hex(canonical_bytes(
                    {"judge_id": j.id, "score": sc}))
                out.append(SignedVerdict(
                    judge_id=j.id, score=sc, rationale=rationale,
                    signature_hex=sig, public_key_hex=s.public_key_hex,
                    algorithm=s.algorithm, key_custody=s.key_custody,
                ))
            else:
                out.append(SignedVerdict(
                    judge_id=j.id, score=sc, rationale=rationale,
                    signature_hex="", public_key_hex="",
                    algorithm="unsigned", key_custody="n/a",
                ))
        return out

    # -- 标定 -----------------------------------------------------------
    def calibrate(self, labeled: Sequence[Tuple[Any, str]],
                  reset_rng: bool = True) -> Dict[str, float]:
        """在带 ground truth 的题集上量化每个裁判的 RMSE。

        labeled : [(puzzle, guess), ...]，ground truth = char_f1(guess, solution)*100

        返回 {judge_id: rmse}。RMSE（而非 MAE）对大偏差更敏感，能更好地把
        「偶尔严重跑偏」的裁判权重压下去。
        """
        if not labeled:
            raise ValueError("标定集不能为空（无 ground truth 则无法标定）")
        if reset_rng:
            for j in self.judges:
                if hasattr(j, "reset_rng"):
                    j.reset_rng()

        err2: Dict[str, List[float]] = {j.id: [] for j in self.judges}
        for puzzle, guess in labeled:
            truth = char_f1(guess, puzzle.solution) * 100.0
            for j in self.judges:
                sc, _ = j.score(puzzle, guess)
                err2[j.id].append((sc - truth) ** 2)

        rmse = {jid: math.sqrt(sum(v) / len(v)) for jid, v in err2.items()}
        self.calibration = dict(rmse)
        return dict(rmse)

    # -- 权重 -----------------------------------------------------------
    @staticmethod
    def weights_from_rmse(rmse: Dict[str, float]) -> Dict[str, float]:
        """RMSE → 共识权重。

        reliability = 1/(rmse+eps)，按最大值归一后套链的 settled-weight 公式：
            w = WEIGHT_BASE + WEIGHT_GAIN * norm   ∈ (1.0, 1.5]

        用**最大值**归一（而非 min-max）是有意的：
          - 裁判同质时所有 norm = 1 ⇒ 所有 w = 1.5 ⇒ R = 1.0 精确成立，
            与「加权 ≡ 等权、无增益」的理论一致；
          - min-max 归一会在同质时 0/0，且一旦有微小噪声就把 R 拉到上界，
            让 R 变成几乎二值的量，失去刻度意义。
        """
        if not rmse:
            raise ValueError("缺少标定数据，无法计算权重")
        rel = {jid: 1.0 / (r + _EPS) for jid, r in rmse.items()}
        rel_max = max(rel.values())
        if rel_max <= 0:
            raise ValueError("标定数据异常：可靠性全为非正")
        return {jid: WEIGHT_BASE + WEIGHT_GAIN * (v / rel_max)
                for jid, v in rel.items()}

    @staticmethod
    def bandwidth_ratio(weights: Dict[str, float]) -> float:
        """R = w_max / w_min。等权时精确为 1.0。"""
        vals = list(weights.values())
        if not vals:
            return 1.0
        lo, hi = min(vals), max(vals)
        if lo <= 0:
            return float("inf")
        return hi / lo

    @staticmethod
    def fault_tolerance(R: float) -> float:
        """闭式拜占庭容忍上限 f_max = R/(R+2)（口径来自该链 README）。"""
        if math.isinf(R):
            return 1.0
        return R / (R + 2.0)

    # -- 共识 -----------------------------------------------------------
    def consensus_verdict(self, puzzle, guess: str,
                          absent: Iterable[str] = (),
                          calibration: Optional[Dict[str, float]] = None,
                          method: str = "weighted") -> ConsensusVerdict:
        """产出面板共识结论。

        method='weighted' 用标定权重；'equal' 为等权对照（用于证明加权是否
        真的带来增益——只报加权结果无法判断增益来源）。
        """
        calib = calibration or self.calibration
        if calib is None:
            raise ValueError("未标定：请先调用 calibrate()（禁止用未标定权重跑共识）")

        weights = self.weights_from_rmse(calib)
        if method == "equal":
            weights = {jid: 1.0 for jid in weights}
        R = self.bandwidth_ratio(weights)

        verdicts = self.score_all(puzzle, guess, absent=absent)
        absent_ids = [jid for jid in self.judge_ids if jid in set(absent)]

        if not verdicts:
            return ConsensusVerdict(
                score=None, accepted=False, verdicts=[], weights=weights,
                R=R, f_max=self.fault_tolerance(R), absent=absent_ids,
                backend="none", method=method,
                detail={"reason": "无任何裁判响应"},
            )

        w_sum = sum(weights[v.judge_id] for v in verdicts)
        weighted = (sum(v.score * weights[v.judge_id] for v in verdicts)
                    / w_sum) if w_sum > 0 else None
        equal = sum(v.score for v in verdicts) / len(verdicts)

        score = weighted if method == "weighted" else equal

        # 共识对象：聚合结论的确定性哈希
        verdict_payload = {
            "scores": {v.judge_id: v.score for v in verdicts},
            "method": method,
            "score": score,
        }
        block_hash = sha256_hex(canonical_bytes(verdict_payload))

        accepted, backend, detail = self._run_consensus(
            block_hash, weights, absent_ids)

        return ConsensusVerdict(
            score=(round(score, 4) if (accepted and score is not None) else None),
            accepted=accepted,
            verdicts=verdicts,
            weights=weights,
            R=R,
            f_max=self.fault_tolerance(R),
            absent=absent_ids,
            backend=backend,
            method=method,
            equal_weight_score=round(equal, 4),
            detail=detail,
        )

    def _run_consensus(self, block_hash: str, weights: Dict[str, float],
                       absent: List[str]) -> Tuple[bool, str, Dict[str, Any]]:
        """跑共识；链不可用则退化为本地权重门限（同一判据，如实标注后端）。"""
        total = sum(weights.values())
        responded = total - sum(weights.get(a, 0.0) for a in absent)
        ratio = responded / total if total > 0 else 0.0
        detail = {
            "block_hash": block_hash,
            "weight_responded": round(responded, 6),
            "weight_total": round(total, 6),
            "responded_ratio": round(ratio, 6),
            "threshold": 2.0 / 3.0,
        }

        if self._pbft is not None:
            try:
                engine = self._pbft["CWPBFTConsensus"](
                    node_id=self.judge_ids[0], consensus_nodes=self.judge_ids)
                for jid, w in weights.items():
                    engine.update_weight(jid, w)
                ok = engine.simulated_consensus(
                    block_hash=block_hash, proposer=self.judge_ids[0],
                    byzantine_nodes=set(absent))
                detail["engine"] = "CWPBFTConsensus.simulated_consensus"
                return ok, "cw-pbft", detail
            except Exception as e:  # 不吞信息：记入 detail 后降级
                detail["engine_error"] = f"{type(e).__name__}: {e}"

        # 降级：同一 2/3 权重门限，本地实现（与链判据一致，便于离线复现）
        ok = ratio > (2.0 / 3.0)
        detail["engine"] = "local-threshold-fallback"
        detail["note"] = ("真实共识引擎不可用，退化为本地 2/3 权重门限；"
                          "判据与 CW-PBFT 一致，但不具备链上记录")
        return ok, "local-weighted", detail
