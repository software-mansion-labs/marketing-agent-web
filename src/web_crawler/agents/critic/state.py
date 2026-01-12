from langchain.agents import AgentState

from web_crawler.agents.output_structures import Critique


class CriticAgentState(AgentState):
    """Extended state of the Agent."""

    website: str
    critique: Critique
