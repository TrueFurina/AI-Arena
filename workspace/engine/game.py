"""海龟汤游戏引擎：状态机 + 回合循环 + 日志输出。对应 specs/engine-design.md。

M0 目标：先用 mock 玩家跑通整条 pipeline（零 key），产出结构化 JSON 日志。
"""

import json
import os
from datetime import datetime

from .roles import Answer


class Game:
    def __init__(self, puzzle, gm, questioner, max_turns=20, log_dir="logs"):
        self.puzzle = puzzle
        self.gm = gm
        self.questioner = questioner
        self.max_turns = max_turns
        self.log_dir = log_dir

        self.phase = "init"          # init -> running -> finished
        self.turn = 0
        self.history = []            # [{"turn", "question", "answer"}]
        self.declared = False
        self.declared_bottom = None

    # ---- 状态机 ----

    def _observation(self):
        """传给提问者的观察（specs §5.1）：仅汤面 + 历史 + 轮次，不含答案。"""
        return {
            "surface": self.puzzle["surface"],
            "history": [
                {"role": "questioner", "content": h["question"]}
                for h in self.history
            ]
            + [{"role": "gm", "answer": h["answer"]} for h in self.history],
            "turn": self.turn,
            "max_turns": self.max_turns,
        }

    def _gm_context(self):
        """GM 私有的真相上下文（不泄漏给提问者）。含 puzzle 本体供 SmartGM 读关键词。"""
        return {
            "surface": self.puzzle["surface"],
            "bottom": self.puzzle["bottom"],
            "key_facts": self.puzzle.get("key_facts", []),
            "clue_gradient": self.puzzle.get("clue_gradient", []),
            "puzzle": self.puzzle,
            "history": self.history,
        }

    def run(self):
        """回合循环，直到 guess 或达到 max_turns。"""
        self.phase = "running"
        while self.phase == "running":
            self.turn += 1
            obs = self._observation()
            action = self.questioner.act(obs)

            if action.get("action") == "guess":
                self.declared = True
                self.declared_bottom = action.get("content", "")
                self.phase = "finished"
                break

            # action == "ask"
            question = action.get("content", "")
            resp = self.gm.answer(question, self._gm_context())
            answer = resp.get("answer", Answer.UNKNOWN.value)
            # M5：GM 可声明本轮回答对（lied），如实记录但不泄漏给提问者
            entry = {"turn": self.turn, "question": question, "answer": answer}
            if resp.get("lied"):
                entry["lied"] = True
            self.history.append(entry)

            if self.turn >= self.max_turns:
                self.phase = "finished"
                break

        return self.build_log()

    # ---- 判定与日志 ----

    def _judge(self):
        """还原度 = 被提问探测到且 GM 确认(yes)的关键事实 / 总关键事实。

        M2 起改为「关键词感知」：命中判定用 puzzle.keywords 做关键词重叠，
        而非要求提问逐字包含事实原文——这样自由改写的提问也能被正确计分。
        """
        key_facts = self.puzzle.get("key_facts", [])
        kw = self.puzzle.get("keywords", {})
        hit = []
        for f in key_facts:
            kws = kw.get(f, [f])  # 无关键词则回退到事实原文
            for h in self.history:
                if h["answer"] == Answer.YES.value and any(k in h["question"] for k in kws):
                    hit.append(f)
                    break
        restoration = (len(hit) / len(key_facts)) if key_facts else 0.0
        return hit, restoration

    @staticmethod
    def _guess_score(declared, bottom):
        """汤底还原保真度：声明汤底与真实汤底的二元字重合度（Jaccard）。
        无需 NLP 依赖，仅用相邻二元组衡量「猜到了多少真相」。"""
        if not declared:
            return 0.0
        def _bigrams(s):
            s = "".join(ch for ch in s if ch.strip())
            return set(s[i:i + 2] for i in range(len(s) - 1))
        b1, b2 = _bigrams(declared), _bigrams(bottom)
        if not b2:
            return 0.0
        return len(b1 & b2) / len(b2)

    def build_log(self):
        hit, restoration = self._judge()
        guess = self._guess_score(self.declared_bottom or "", self.puzzle.get("bottom", ""))
        efficiency = round(restoration / self.turn, 4) if self.turn else 0.0
        # M5：统计 GM 实际撒谎比例（向后兼容：旧日志无 lied 字段则记 0）
        # 分母用「已回答轮数」len(history)，而非 self.turn（含最后的 guess 轮，无 GM 回答）
        lies_total = sum(1 for h in self.history if h.get("lied"))
        answered = len(self.history)
        lie_rate_actual = round(lies_total / answered, 4) if answered else 0.0
        return {
            "meta": {
                "model": getattr(self.questioner, "name", "mock"),
                "puzzle_id": self.puzzle["id"],
                "mode": "single",
                "max_turns": self.max_turns,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            "surface": self.puzzle["surface"],
            "bottom": self.puzzle["bottom"],
            "key_facts": self.puzzle.get("key_facts", []),
            "turns": self.history,
            "result": {
                "declared": self.declared,
                "declared_bottom": self.declared_bottom,
                "restoration": round(restoration, 4),
                "guess_score": round(guess, 4),
                "efficiency": efficiency,
                "turns_used": self.turn,
                "key_facts_hit": hit,
                "deception": {
                    "lies_total": lies_total,
                    "lie_rate": lie_rate_actual,
                },
            },
        }

    def save(self, path=None):
        os.makedirs(self.log_dir, exist_ok=True)
        if path is None:
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            # 模型名可能含 "/"（如 litellm 的 deepseek/deepseek-chat），
            # 落盘文件名需做安全化，避免被解析成嵌套路径；meta.model 保留原值。
            model = getattr(self.questioner, "name", "mock").replace("/", "_")
            path = os.path.join(self.log_dir, f"{model}_{self.puzzle['id']}_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.build_log(), f, ensure_ascii=False, indent=2)
        return path
