"""M3 多模型对战编排（零 key 可跑）。

读 config.toml 的 [[models]] 注册表，对每个 enabled 模型 × 每题跑 N 局（vs SmartGM），
日志落 logs/，结束可选调用 evaluate.py 聚合出报告。

- 真实模型：litellm 名非空，靠 .env 的 key 自动接入（enabled 需手动开）。
- mock 模型：mock="strong"/"weak"，零 key 即可跑，用于验证编排与指标区分度。
- --mock：把所有真实模型临时替换成 mock-strong，纯粹验证零 key 编排。

用法：
  python run/tournament.py                  # 按 config.toml 跑全部 enabled
  python run/tournament.py --mock          # 零 key：真实模型也用 mock 跑，验证编排
  python run/tournament.py --model deepseek --puzzle turtle_001 --runs 2
  python run/tournament.py --no-evaluate    # 不自动聚合
  python run/tournament.py --config my.toml # 指定注册表
"""

import argparse
import os
import subprocess
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from engine.game import Game
from engine.puzzles import PUZZLES, get_puzzle
from players.mock_player import MockQuestioner, WeakMockQuestioner
from players.gm import SmartGM
from players.mock_player import MockGM
from players.llm_player import LLMQuestioner


def _build_gm(kind):
    return SmartGM() if kind == "smart" else MockGM()


def _build_questioner(entry, force_mock, max_turns):
    """根据注册表条目构造提问者。force_mock=True 时真实模型也用 mock-strong。"""
    mock = entry.get("mock")
    litellm_name = entry.get("litellm")
    if mock:
        if mock == "weak":
            return "mock-weak", WeakMockQuestioner
        return "mock-strong", MockQuestioner
    if force_mock or not litellm_name:
        return "mock-strong", MockQuestioner
    # 真实模型：用完整 litellm 名或别名
    return litellm_name, lambda *a, **k: LLMQuestioner.from_model(
        litellm_name, max_questions=max_turns)


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI-Arena M3 多模型对战编排")
    parser.add_argument("--config", default="config.toml", help="模型注册表路径")
    parser.add_argument("--mock", action="store_true",
                        help="零 key：把所有真实模型临时替换为 mock-strong 验证编排")
    parser.add_argument("--model", default=None, help="只跑某个 alias")
    parser.add_argument("--puzzle", default=None, help="只跑某题 id（缺省全部题）")
    parser.add_argument("--runs", type=int, default=None, help="覆盖每题局数")
    parser.add_argument("--no-evaluate", action="store_true", help="跑完不自动聚合")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise SystemExit(f"❌ 找不到注册表: {args.config}")
    cfg = load_config(args.config)
    g = cfg.get("globals", {})
    max_turns = int(g.get("max_turns", 20))
    runs = int(args.runs or g.get("runs_per_puzzle", 2))
    gm_kind = g.get("gm", "smart")
    log_dir = g.get("log_dir", "logs")
    evaluate_after = g.get("evaluate_after", True) and not args.no_evaluate

    models = [m for m in cfg.get("models", []) if m.get("enabled", False)]
    if args.model:
        models = [m for m in models if m.get("alias") == args.model]
    if not models:
        raise SystemExit("❌ 没有 enabled 的模型（真实模型需在 config.toml 置 enabled=true 并填 .env key）。")

    puzzles = [get_puzzle(args.puzzle)] if args.puzzle else list(PUZZLES)
    gm = _build_gm(gm_kind)

    print("=" * 64)
    print("AI-Arena · M3 多模型对战")
    print(f"  模式: {'零key(mock)' if args.mock else '按注册表'}  "
          f"GM={gm_kind}  runs/题={runs}  题目数={len(puzzles)}  模型数={len(models)}")
    print("=" * 64)

    total = 0
    for m in models:
        alias = m.get("alias")
        label, qcls = _build_questioner(m, args.mock, max_turns)
        print(f"\n▶ 模型 {alias}  (执行形态: {label})")
        for p in puzzles:
            for r in range(1, runs + 1):
                if qcls is WeakMockQuestioner:
                    q = WeakMockQuestioner(p["key_facts"], p["bottom"], max_questions=max_turns)
                elif qcls is MockQuestioner:
                    q = MockQuestioner(p["key_facts"], p["bottom"], max_questions=max_turns)
                else:
                    q = qcls()  # LLMQuestioner.from_model 已实例化
                game = Game(p, gm, q, max_turns=max_turns, log_dir=log_dir)
                try:
                    log = game.run()
                except Exception as e:
                    print(f"   ✗ {p['id']} 第{r}局失败: {e}")
                    continue
                path = game.save()
                total += 1
                res = log["result"]
                print(f"   · {p['id']} 第{r}局  还原度={res['restoration']}  "
                      f"汤底保真={res['guess_score']} 轮次={res['turns_used']}  -> {os.path.basename(path)}")

    print("\n" + "=" * 64)
    print(f"完成：共 {total} 局，日志在 {log_dir}/")
    if evaluate_after:
        print("正在聚合评测报告...")
        try:
            subprocess.run([sys.executable, "run/evaluate.py"],
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           check=True)
        except Exception as e:
            print(f"⚠️ 自动聚合失败（可手动跑 python run/evaluate.py）: {e}")
    else:
        print("未自动聚合；需报告时跑：python run/evaluate.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
