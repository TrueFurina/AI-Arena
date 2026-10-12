"""为对局日志签发链上评测凭证（P1 的命令行入口）。

用法：
    python issue_credential.py logs/game_xxx.json              # 单个
    python issue_credential.py logs/*.json --out logs/credentials
    python issue_credential.py logs/*.json --no-chain          # 强制仅本地签名

输出：每份日志一个 <name>.credential.json，并逐条打印验签结论。
退出码：全部验签通过为 0，任一失败为 1（便于在流水线里做门禁）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chain.recorder import MatchRecorder
from chain.verify import verify_credential


def main() -> int:
    ap = argparse.ArgumentParser(description="为对局日志签发链上评测凭证")
    ap.add_argument("logs", nargs="+", help="对局日志 JSON（GameResult 格式）")
    ap.add_argument("--out", default="logs/credentials", help="凭证输出目录")
    ap.add_argument("--no-chain", action="store_true",
                    help="不提交共识链，仅本地签名（用于离线/降级验证）")
    ap.add_argument("--chain-path", default=None,
                    help="真实共识链路径（默认取 ARENA_CHAIN_PATH 或内置默认）")
    ap.add_argument("--judge-id", default="default-judge")
    args = ap.parse_args()

    out_dir = Path(args.out)
    rec = MatchRecorder(chain_path=args.chain_path,
                        submit_to_chain=not args.no_chain)
    print(f"[issue] 链可用={rec._chain is not None} "
          f"| 使用链crypto={rec.signer.using_chain_crypto} "
          f"| 密钥保管={rec.signer.key_custody}")

    ok_all = True
    for log_path in args.logs:
        p = Path(log_path)
        if not p.exists():
            print(f"  ⚠️ 跳过（不存在）: {p}")
            continue
        try:
            res = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  ⚠️ 跳过（解析失败 {type(exc).__name__}）: {p}")
            continue

        cred = rec.issue_from_result(res, judge_id=args.judge_id)
        vr = verify_credential(cred)
        saved = rec.save(cred, out_dir / f"{p.stem}.credential.json")
        anchor = cred.anchor["status"] if cred.anchor else "未上链(仅本地签名)"
        print(f"  {'✅' if vr.ok else '❌'} {p.name} -> {saved.name} "
              f"| 锚定={anchor} | {vr.summary()}")
        ok_all = ok_all and vr.ok

    print("\n[issue] 全部凭证验签通过 ✅" if ok_all
          else "\n[issue] 存在验签失败的凭证 ❌")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
