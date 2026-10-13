"""P3 · 报告渲染（把流水线产物写成 markdown）。

刻意与 `pipeline.py` 的编排分离：渲染是纯函数（dict → str），不含任何
指标计算，**也不含任何 fallback 逻辑**。

为什么单独成模块（不只是代码整洁）：
  报告是本项目对外的叙述面。「共识未发布」如果在渲染层被写成「0 分」或
  悄悄补一个单点分数，P2-A 的内核价值（拒绝发布）就在这里被无声抹掉——
  这类「静默填补」正是本项目反复栽过的坑（风格名冒号解析、λ 惰性）。
  把渲染独立出来，就能对它单独做变异验证（见 tests/test_pipeline.py）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# 报告头部固定声明的能力边界（对外叙述不得省略，与 chain/__init__.py 同源）
BOUNDARY_NOTES: List[str] = [
    "本凭证锚定的是**本地共识链**，非公链；`in-pool` 表示已进交易池但未出块，不具最终性。",
    "验签通过只证明「凭证内记录自签发以来未被篡改」；凭证自带公钥，"
    "**未接身份锚定时，任何人都能自造密钥签发自洽的假凭证且验签通过**。",
    "启用共识时，面板裁判是 `judges/biased.py` 的**测量误差模型**，"
    "不是真实 LLM 裁判——受控结论不可外推为真实裁判的聚合增益。",
    "共识结论与客观指标一同写入被签名的记录；但它回答的是「结论是否可信」，"
    "不回答「签发者是谁」（后者需身份锚定，属部署问题）。",
]


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_report(artifacts: Dict[str, Any]) -> str:
    """把流水线产物渲染成 markdown。

    artifacts 必需键：game / metrics；可选：consensus / credential / verify /
    credential_path / game_path / degraded / identity_used。
    缺失的可选键一律渲染为「未执行」，**不编造占位数字**。
    """
    game: Dict[str, Any] = artifacts["game"]
    metrics: Dict[str, Any] = artifacts["metrics"]
    cons: Optional[Dict[str, Any]] = artifacts.get("consensus")
    cred: Optional[Dict[str, Any]] = artifacts.get("credential")
    ver: Optional[Dict[str, Any]] = artifacts.get("verify")
    degraded: List[str] = list(artifacts.get("degraded", []))

    lines: List[str] = []
    lines.append(f"# AI-Arena 评测报告 · `{game.get('game_id', 'unknown')}`")
    lines.append("")
    lines.append(f"- 题目：`{game.get('puzzle', {}).get('soup_id', 'unknown')}`"
                 f"（{game.get('puzzle', {}).get('title', '')}）")
    lines.append(f"- 主持：{game.get('host_model', 'unknown')} ｜ "
                 f"提问者：{', '.join(game.get('players', []) or []) or '—'}")
    lines.append(f"- 轮数：{game.get('max_rounds', '—')} ｜ "
                 f"胜者（引擎内单点裁判口径）：{game.get('winner') or '—'}")
    lines.append("")

    # ---- 1 客观指标 -----------------------------------------------------
    lines.append("## 1. 客观指标（唯一口径：`engine/evaluate.py`）")
    lines.append("")
    lines.append("| 指标 | 值 | 含义 |")
    lines.append("|---|---|---|")
    meaning = {
        "valid_q": "有效提问数",
        "total_q": "提问总数",
        "vra": "有效提问率",
        "yn_rate": "是/否作答率",
        "hit_rate": "命中率（含重复问）",
        "dhit": "去重命中率",
        "f1": "终局还原 char-F1",
    }
    for k in ("valid_q", "total_q", "vra", "yn_rate", "hit_rate", "dhit", "f1"):
        if k in metrics:
            lines.append(f"| `{k}` | {_fmt(metrics[k])} | {meaning.get(k, '')} |")
    if metrics.get("guess"):
        lines.append("")
        lines.append("终局还原（截断）：")
        lines.append("")
        lines.append("```")
        lines.append(str(metrics["guess"]))
        lines.append("```")
    lines.append("")

    # ---- 2 裁判结论 -----------------------------------------------------
    lines.append("## 2. 裁判结论")
    lines.append("")
    lines.append("### 2.1 引擎内单点裁判（仅参考，不作结论）")
    lines.append("")
    scores = game.get("scores") or []
    if scores:
        lines.append("| 提问者 | 分数 |")
        lines.append("|---|---|")
        for s in scores:
            lines.append(f"| {s.get('player', '—')} | {_fmt(s.get('score'))} |")
    else:
        lines.append("（本局未配置引擎内裁判）")
    lines.append("")
    lines.append("> 续5 实测：LLM 裁判自评 score 与客观 f1 系统性不一致（自偏好），"
                 "故单点分数仅作参考。")
    lines.append("")

    lines.append("### 2.2 多裁判共识（P2-A）")
    lines.append("")
    if cons is None:
        lines.append("未启用共识面板（本次为单点裁判路径）。")
    else:
        accepted = bool(cons.get("accepted"))
        lines.append("| 项 | 值 |")
        lines.append("|---|---|")
        lines.append(f"| 共识状态 | {'✅ 已发布' if accepted else '⛔ 未发布（REJECTED）'} |")
        lines.append(f"| 共识结论分数 | {_fmt(cons.get('score'))} |")
        lines.append(f"| 等权对照 | {_fmt(cons.get('equal_weight_score'))} |")
        lines.append(f"| 权重带宽比 R | {_fmt(cons.get('R'))} |")
        lines.append(f"| 拜占庭容忍上限 f_max | {_fmt(cons.get('f_max'))} |")
        lines.append(f"| 共识后端 | {cons.get('backend', '—')} |")
        lines.append(f"| 缺席裁判 | {', '.join(cons.get('absent', []) or []) or '（无）'} |")
        lines.append("")
        if not accepted:
            lines.append("**共识拒绝发布结论**（可靠权重不足或裁判缺席）。"
                         "按设计判断 1，本报告**不提供任何替代分数**——"
                         "「拒绝给数」正是共识层相对单点裁判的价值所在，"
                         "用单点分数填补等于把内核退回装饰。")
        else:
            lines.append(f"权重：{cons.get('weights')}")
        lines.append("")
    lines.append("")

    # ---- 3 凭证 ---------------------------------------------------------
    lines.append("## 3. 链上凭证（P1 · 完整性 + P2-A · 共识进签名）")
    lines.append("")
    if cred is None:
        lines.append("未签发凭证。")
    else:
        record = cred.get("record", {}) or {}
        anchor = cred.get("anchor") or None
        lines.append("| 项 | 值 |")
        lines.append("|---|---|")
        lines.append(f"| 记录摘要 `record_hash` | `{cred.get('record_hash', '—')}` |")
        lines.append(f"| 凭证 ID | `{str(cred.get('record_hash', ''))[:16] or '—'}` |")
        lines.append(f"| 签名者 | {cred.get('signer_id', '—')} |")
        lines.append(f"| 算法 | {cred.get('algorithm', '—')} |")
        lines.append(f"| 密钥保管 | {cred.get('key_custody', '—')} |")
        lines.append(f"| 后端 | {cred.get('backend', '—')} |")
        lines.append(f"| 锚定状态 | {anchor.get('status') if anchor else '未上链（仅本地签名）'} |")
        lines.append(f"| `tx_id` | {anchor.get('tx_id') if anchor else '—'} |")
        lines.append(f"| 共识已进签名对象 | "
                     f"{'是' if record.get('verdict', {}).get('mode') == 'panel-consensus' else '否（单点路径）'} |")
        lines.append("")
        if ver is not None:
            steps = ver.get("steps", {}) or {}
            lines.append(f"验签结论：**{'✅ 可信' if ver.get('ok') else '❌ 不可信'}**")
            lines.append("")
            for k, v in steps.items():
                lines.append(f"- {k}：{'通过' if v else '失败'}")
            if ver.get("identity_status"):
                lines.append(f"- identity：{ver['identity_status']}"
                             f"（{'已做' if ver['identity_status'] != 'unavailable' else '合约不可用，未参与判定'}）")
            else:
                lines.append("- identity：未执行（未接入身份锚定，见能力边界第 2 条）")
            if ver.get("reasons"):
                for r in ver["reasons"].values():
                    lines.append(f"  - {r}")
        lines.append("")
        lines.append("验签入口（任选其一，均无需访问本项目运行环境）：")
        lines.append("")
        lines.append("```bash")
        lines.append(f"python web_app.py          # → http://localhost:8860/verify")
        lines.append(f"python issue_credential.py {artifacts.get('game_path', 'logs/')}")
        lines.append("```")
        lines.append("")
    lines.append("")

    # ---- 4 降级与边界 ---------------------------------------------------
    if degraded:
        lines.append("## 4. 本次运行的降级记录（如实，不静默）")
        lines.append("")
        for d in degraded:
            lines.append(f"- ⚠️ {d}")
        lines.append("")
        n = 5
    else:
        n = 4

    lines.append(f"## {n}. 能力边界（对外叙述不得省略）")
    lines.append("")
    for b in BOUNDARY_NOTES:
        lines.append(f"- {b}")
    lines.append("")

    lines.append(f"## {n + 1}. 产物清单")
    lines.append("")
    for label, key in (("对局日志", "game_path"), ("凭证", "credential_path")):
        p = artifacts.get(key)
        lines.append(f"- {label}：`{p}`" if p else f"- {label}：未落盘")
    lines.append("")
    return "\n".join(lines)
