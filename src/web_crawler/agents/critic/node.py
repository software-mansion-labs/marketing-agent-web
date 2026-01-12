from enum import Enum

from langgraph.graph import START, END


class CriticAgentNode(str, Enum):
    DESCRIPTION = "DESCRIPTION"
    INTRODUCTION = "INTRODUCTION"
    CRITIQUE = "CRITIQUE"
    START = START
    END = END
