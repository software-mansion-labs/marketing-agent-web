from enum import Enum

from langgraph.graph import END, START


class SearchAgentNode(str, Enum):
    DESCRIPTION = "DESCRIPTION"
    SEARCH = "SEARCH"
    TOOLS_SEARCHER = "TOOLS_SEARCHER"
    SELECT_PAGE = "SELECT_PAGE"
    LOAD = "LOAD"
    CRITIQUE = "CRITIQUE"
    SUMMARY = "SUMMARY"
    START = START
    END = END
