"""编排一局海龟汤。
- M0：--mode mock（零 key）跑通整条 pipeline。
- M1：--mode llm（litellm 真实模型当提问者）。
- M2：--gm {mock,smart} 选择主持人；默认 smart（关键词感知，让还原度可信）。

无 key / 只想看提示词时，可加 --dry-run 仅打印将要发给模型的消息，不发请求。
"""

import argparse
import os
import sys

# 让脚本可从 /workspace 直接运行，导入 engine/players 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import Game
from engine.puzzles import get_puzzle
from players.mock_player import MockGM, MockQuestioner
from players.gm import SmartGM
from players.llm_player import LLMQuestioner


def build_questioner(mode, model, max_turns):
    if mode == "mock":
        return None  # 由调用方用 puzzle 构造 MockQuestioner
    if mode == "llm":
        if not model:
            raise SystemExit("❌ llm 模式需要 --model（如 deepseek / qwen / kimi / gpt / claude，或完整 litellm 名）。")
        return LLMQuestioner.from_model(model, max_questions=max_turns)


def build_gm(kind):
    return SmartGM() if kind == "smart" else MockGM()


def main():
    parser = argparse.ArgumentParser(description="AI-Arena 单局海龟汤")
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock",
                        help="mock=规则化假玩家(零key)；llm=真实模型(M1 接入 litellm)")
    parser.add_argument("--model", default=None, help="llm 模式下的模型（别名或完整 litellm 名）")
    parser.add_argument("--gm", choices=["mock", "smart"], default="smart",
                        help="主持人类型：mock=逐字匹配；smart=关键词感知(M2，默认)")
    parser.add_argument("--puzzle", default=None, help="题目 id，缺省随机")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--dry-run", action="store_true",
                        help="llm 模式专用：只打印提示词与用户消息，不调用 API（无需 key）")
    args = parser.parse_args()

    puzzle = get_puzzle(args.puzzle)
    gm = build_gm(args.gm)

    # ---- 构造提问者 ----
    if args.mode == "mock":
        questioner = MockQuestioner(puzzle["key_facts"], puzzle["bottom"], max_questions=args.max_turns)
    else:  # llm
        questioner = build_questioner(args.mode, args.model, args.max_turns)
        if args.dry_run:
            # 仅预览首轮提示词，不发请求
            game = Game(puzzle, gm, questioner, max_turns=args.max_turns, log_dir=args.log_dir)
            obs = game._observation()
            print(f"=== DRY RUN · 模型 {questioner.model} · GM {gm.name} · 题目 {puzzle['id']} ===\n")
            questioner.preview(obs)
            print("\n（未发送任何请求。填好 .env 后去掉 --dry-run 即可真实对局。）")
            return

    game = Game(puzzle, gm, questioner, max_turns=args.max_turns, log_dir=args.log_dir)
    try:
        log = game.run()
    except Exception as e:  # 真实模型调用可能遇鉴权/网络/限流错误
        print(f"❌ 真实模型调用失败：{e}")
        print("   请确认：① 已在 .env 填入对应供应商的 API key（见 .env.example）；")
        print("           ② 网络可直达该供应商端点（必要时配置 LITELLM_PROXY_URL）。")
        raise SystemExit(1)
    path = game.save()

    print(f"✅ 一局完成（{args.mode} 模式 · GM={gm.name}）")
    print(f"   模型 : {log['meta']['model']}")
    print(f"   题目 : {puzzle['id']}")
    print(f"   轮数 : {log['result']['turns_used']}")
    print(f"   还原度: {log['result']['restoration']}")
    print(f"   汤底保真: {log['result']['guess_score']}")
    print(f"   命中关键事实: {log['result']['key_facts_hit']}")
    print(f"   日志 : {path}")


if __name__ == "__main__":
    main()
