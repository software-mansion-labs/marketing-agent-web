from typing import Literal

from pydantic import BaseModel, Field

from web_crawler.agents.output_structures import WebsiteHeader
from web_crawler.agents.search import SearchAgentNode


class WebsitesToLoad(BaseModel):
    """Websites to load."""

    websites: list[WebsiteHeader] = Field(description="list of websites to load")


class LoopDecision(BaseModel):
    """Decision on the next action to take in loop."""

    loop_decision: Literal[SearchAgentNode.SEARCH, SearchAgentNode.SUMMARY] = Field(
        description="whether to keep searching or end the workflow"
    )
