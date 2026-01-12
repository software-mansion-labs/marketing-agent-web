from enum import Enum

from langgraph.graph import START, END


class SelectorAgentNode(str, Enum):
    DESCRIPTION = "DESCRIPTION"
    INTRODUCTION = "INTRODUCTION"
    SELECTION = "SELECTION"
    START = START
    END = END
