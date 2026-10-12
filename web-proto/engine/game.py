"""海龟汤（Turtle Soup）对局引擎。

核心模型：
- Puzzle：一道题，含汤面(surface)、汤底(solution)，可选 golden_qa 供 mock 主持回答。
- Message：对局中的一条发言（提问/回答/终局还原/系统）。
- TurtleSoupGame：主持(host)知道汤底，多个提问者(questioner)轮流问 yes/no 问题，
  到达 max_rounds 后各自写出最终还原，由 judge 打分。

设计原则（先跑通再优化）：
- 纯 Python、无外部依赖即可用 Mock 玩家跑完整流程；
- 真实模型通过 players/llm.py 的 litellm 适配层接入，不影响引擎。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Puzzle:
    soup_id: str
    title: str
    surface: str           # 汤面：抛给玩家的谜面
    solution: str          # 汤底：完整真相
    golden_qa: List[dict] = field(default_factory=list)  # [{contains, answer}] mock 主持用
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Message:
    round: int
    speaker: str           # player id
    role: str              # question | answer | guess | system
    content: str
    ts: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlayerScore:
    player: str
    model: str
    final_guess: str
    score: float
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GameResult:
    game_id: str
    puzzle: dict
    host_model: str
    players: List[str]
    max_rounds: int
    transcript: List[dict]
    scores: List[dict]
    winner: Optional[str]
    started_at: float
    ended_at: float

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class TurtleSoupGame:
    """海龟汤一局对战的主循环。"""

    def __init__(
        self,
        puzzle: Puzzle,
        host,                       # Host 实例
        questioners: List,         # Player 实例列表
        judge=None,                # 打分器，可选
        max_rounds: int = 8,
    ):
        self.puzzle = puzzle
        self.host = host
        self.questioners = questioners
        self.judge = judge
        self.max_rounds = max_rounds
        self.transcript: List[Message] = []
        self.game_id = uuid.uuid4().hex[:8]
        self.started_at = 0.0
        self.ended_at = 0.0

    # ---- 内部工具 ----
    def _emit(self, round_: int, speaker: str, role: str, content: str) -> Message:
        m = Message(round_, speaker, role, content, time.time())
        self.transcript.append(m)
        return m

    def _transcript_dicts(self) -> List[dict]:
        return [m.to_dict() for m in self.transcript]

    # ---- 主流程 ----
    def run(self) -> GameResult:
        self.started_at = time.time()
        self._emit(0, "system", "system", f"【汤面】{self.puzzle.surface}")

        for r in range(1, self.max_rounds + 1):
            for q in self.questioners:
                question = q.ask_question(self.puzzle, self._transcript_dicts())
                self._emit(r, q.id, "question", question)
                answer = self.host.answer(self.puzzle, question, self._transcript_dicts())
                self._emit(r, self.host.id, "answer", answer)

        # 终局：每个提问者写出最终还原，judge 打分
        scores: List[PlayerScore] = []
        for q in self.questioners:
            guess = q.final_guess(self.puzzle, self._transcript_dicts())
            self._emit(self.max_rounds + 1, q.id, "guess", guess)
            if self.judge is not None:
                sc, rationale = self.judge.score(self.puzzle, guess)
            else:
                sc, rationale = 0.0, "（无 judge，未评分）"
            scores.append(PlayerScore(q.id, q.model, guess, sc, rationale))

        self.ended_at = time.time()

        # 胜负：分数最高者胜（并列取最先）
        winner = None
        if scores:
            best = max(scores, key=lambda s: s.score)
            winner = best.player if best.score > 0 else None

        result = GameResult(
            game_id=self.game_id,
            puzzle=self.puzzle.to_dict(),
            host_model=self.host.model,
            players=[q.id for q in self.questioners],
            max_rounds=self.max_rounds,
            transcript=self._transcript_dicts(),
            scores=[s.to_dict() for s in scores],
            winner=winner,
            started_at=self.started_at,
            ended_at=self.ended_at,
        )
        return result
