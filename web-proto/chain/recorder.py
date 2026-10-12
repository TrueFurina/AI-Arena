"""评测凭证签发（P1）—— 把一局对局结论变成可离线验证的凭证。

口径纪律：本模块**不计算任何指标**，全部来自 `engine/evaluate.py`
（`metrics_from_result`）。历史上指标口径曾出现三份拷贝并存、改动一处即让
历史结论失效的事故，故此处只做搬运与组装。

锚定纪律：`anchor` 里的 `status` 如实区分两种状态——
  - `"in-pool"`：交易已进共识链交易池，但**尚未出块**，不具最终性
  - `"anchored"`：已落块
不允许把入池写成「已上链」。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from engine.evaluate import metrics_from_result

from .crypto_adapter import Signer, try_import_marl_chain
from .models import Credential, EvaluationRecord, canonical_bytes, sha256_hex


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def transcript_digest(transcript) -> str:
    """transcript 的 sha256：保证原始对话不可篡改（改动即哈希不符）。"""
    payload = [
        {
            "round": m.get("round"),
            "speaker": m.get("speaker"),
            "role": m.get("role"),
            "content": m.get("content", ""),
        }
        for m in transcript
    ]
    return sha256_hex(canonical_bytes({"transcript": payload}))


class MatchRecorder:
    """签发评测凭证。"""

    def __init__(self,
                 signer: Optional[Signer] = None,
                 chain_path: Optional[str] = None,
                 submit_to_chain: bool = True,
                 key_dir: Optional[Path] = None):
        self.signer = signer or Signer(key_dir=key_dir, chain_path=chain_path)
        self.chain_path = chain_path
        self.submit_to_chain = submit_to_chain
        self._chain = try_import_marl_chain(chain_path)
        self._ledger = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # 记录构造
    # ------------------------------------------------------------------
    def build_record(self, res: Dict[str, Any],
                     judge_id: str = "default-judge") -> EvaluationRecord:
        metrics = metrics_from_result(res)
        puzzle = res.get("puzzle", {}) or {}
        scores = res.get("scores", []) or []

        participants: Dict[str, str] = {"host": str(res.get("host_model", "unknown"))}
        for p in res.get("players", []) or []:
            participants.setdefault(f"player_{len(participants)}", str(p))

        verdict = {
            "judge_id": judge_id,
            "winner": res.get("winner"),
            "scores": [
                {
                    "player": s.get("player"),
                    "model": s.get("model"),
                    "score": s.get("score"),
                }
                for s in scores
            ],
        }

        return EvaluationRecord(
            match_id=str(res.get("game_id", "unknown")),
            puzzle_id=str(puzzle.get("soup_id", puzzle.get("id", "unknown"))),
            participants=participants,
            rounds=int(res.get("max_rounds", 0) or 0),
            metrics={
                k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float))
            },
            transcript_hash=transcript_digest(res.get("transcript", [])),
            verdict=verdict,
            created_at=_iso_now(),
        )

    # ------------------------------------------------------------------
    # 签发
    # ------------------------------------------------------------------
    def issue(self, record: EvaluationRecord,
              submit: Optional[bool] = None) -> Credential:
        payload = record.to_payload()
        message = record.canonical_bytes()
        signature_hex = self.signer.sign_hex(message)

        anchor = None
        do_submit = self.submit_to_chain if submit is None else submit
        if do_submit:
            anchor = self._submit_to_chain(payload, signature_hex)

        return Credential(
            record=payload,
            record_hash=sha256_hex(message),
            signature_hex=signature_hex,
            public_key_hex=self.signer.public_key_hex,
            signer_id=self.signer.signer_id,
            algorithm=self.signer.algorithm,
            key_custody=self.signer.key_custody,
            anchor=anchor,
            backend=("marl-ecdsa" if self.signer.using_chain_crypto else "local"),
        )

    def issue_from_result(self, res: Dict[str, Any],
                          judge_id: str = "default-judge") -> Credential:
        return self.issue(self.build_record(res, judge_id=judge_id))

    # ------------------------------------------------------------------
    # 上链（失败降级，绝不阻塞评测）
    # ------------------------------------------------------------------
    def _submit_to_chain(self, payload: Dict[str, Any],
                         signature_hex: str) -> Optional[Dict[str, Any]]:
        if not self._chain:
            self.last_error = "真实共识链不可用（未走链，仅本地签名）"
            return None
        try:
            Blockchain = self._chain["Blockchain"]
            Transaction = self._chain["Transaction"]
            ECDSAUtils = self._chain["ECDSAUtils"]

            if self._ledger is None:
                persist = Path(self.signer.key_dir) / "arena_chain.json"
                self._ledger = Blockchain(persist_path=str(persist))

            message = canonical_bytes(payload)
            tx = Transaction(
                tx_id=ECDSAUtils.hash_message(message).hex()[:32],
                agent_id=self.signer.signer_id,
                action={"kind": "arena-evaluation", "schema": payload.get("schema")},
                action_hash=ECDSAUtils.hash_message(message).hex(),
                timestamp=int(time.time() * 1000),
                nonce=int(time.time() * 1000),
                signature_hex=signature_hex,
                tx_type="score",
                extra={"credential_payload": payload},
            )
            ok = self._ledger.add_transaction(tx)
            if not ok:
                self.last_error = "add_transaction 返回 False"
                return None
            return {
                "chain": "marl-ecdsa-consensus",
                "tx_id": tx.tx_id,
                # 入池 ≠ 出块：如实标注，不写成「已上链」
                "status": "in-pool",
                "pool_size": self._ledger.get_tx_pool_size(),
            }
        except Exception as exc:  # 链异常不得阻断评测
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    # ------------------------------------------------------------------
    def save(self, credential: Credential, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(credential.to_json(), encoding="utf-8")
        return p
