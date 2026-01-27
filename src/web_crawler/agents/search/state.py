from langchain.agents import AgentState

from web_crawler.agents.output_structures import (
    Website,
    WebsiteChoiceList,
    WebsiteCritique,
)
from web_crawler.agents.search.output_structures import WebsitesToLoad


class SearchAgentState(AgentState):
    """Extended Agent state."""

    id: int
    search_loop_iteration: int
    websites_to_load: WebsitesToLoad
    loaded_websites: list[Website]
    website_critiques: list[WebsiteCritique]
    selection: WebsiteChoiceList
