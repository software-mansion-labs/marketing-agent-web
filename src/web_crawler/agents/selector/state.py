from langchain.agents import AgentState

from web_crawler.agents.output_structures import (
    WebsiteChoiceList,
    WebsiteCritique,
)


class SelectorAgentState(AgentState):
    """Extended state of the Agent."""

    website_critiques: list[WebsiteCritique]
    selection: WebsiteChoiceList
