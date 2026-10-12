"""角色定义与答案词表。对应 specs/engine-design.md §4/§5.3。"""

from enum import Enum


class Role(str, Enum):
    GM = "gm"
    QUESTIONER = "questioner"
    # LIAR = "liar"  # M5 骗子变体预留


class Stance(str, Enum):
    """M6 辩论立场。"""
    PRO = "pro"
    CON = "con"
    # JUDGE = "judge"  # 裁判不参与立场


class Answer(str, Enum):
    """GM 是非回答词表（specs §5.3）。"""

    YES = "yes"
    NO = "no"
    IRRELEVANT = "irrelevant"
    YES_AND_NO = "yes_and_no"
    UNKNOWN = "unknown"


ANSWER_VALUES = [a.value for a in Answer]
