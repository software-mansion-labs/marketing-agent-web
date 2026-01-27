import logging

import requests
from bs4 import BeautifulSoup
from langchain.tools import BaseTool
from langgraph.prebuilt import ToolNode
from more_itertools import unique_everseen
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from web_crawler.agents.base_agent import BaseAgent
from web_crawler.agents.critic.agent import CriticAgent
from web_crawler.agents.search import SearchAgentNode, SearchAgentState
from web_crawler.agents.selector.agent import SelectorAgent
from web_crawler.agents.output_structures import Website, WebsiteChoice
from web_crawler.agents.search.output_structures import WebsitesToLoad, LoopDecision

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent[SearchAgentState]):
    """AI agent meant to search the Web for marketing purposes."""

    def __init__(
        self,
        search_tool: BaseTool,
        critic: CriticAgent,
        selector: SelectorAgent,
        description_prompt: str,
        search_prompt: str,
        select_page_prompt: str,
        decide_loop_prompt: str,
        model: str = "openai:gpt-4o",
        min_iterations: int = 2,
        max_iterations: int = 5,
    ) -> None:
        """Initializes the Agent's workflow graph and LLM model.

        Args:
            search_tool (Tool): tool to use to search for websites.
            model (str, optional): LLM model to use as foundation for agents. Defaults to "openai:gpt-4o".
            critic (CriticAgent): agent that critiques website candidates.
            selector (SelectorAgent): agent that selects best candidates.
            description_prompt (str): description of the product.
            search_prompt (str): prompt used to search for websites.
            select_page_prompt (str): prompt used to select pages to visit.
            decide_loop_prompt (str): prompt used to decide on loop.
            min_iterations (int, optional): minimum number of iterations of the search loop. Defaults to 2.
            max_iterations (int, optional): maximum number of iterations of the search loop. Defaults to 5.
        """
        assert (
            min_iterations <= max_iterations
        ), "min_iterations must be smaller than max_iterations"
        super().__init__(model)
        self._search_tool = search_tool
        self._min_iterations = min_iterations
        self._max_iterations = max_iterations
        self._critic = critic
        self._selector = selector
        self._description_prompt = description_prompt
        self._search_prompt = search_prompt
        self._select_page_prompt = select_page_prompt
        self._decide_loop_prompt = decide_loop_prompt
        self._workflow = self._build_workflow()

    def run(self, tries: int = 1) -> list[WebsiteChoice]:
        """Runs the Agent.

        Args:
            tries (int, optional): how many times to run the agent. Defaults to 1.

        Returns:
            list[WebsiteChoice]: suitable websites and justifications for their suitability.
        """
        logger.info("Running the Agent.")

        inputs = [
            {
                "id": id,
                "search_loop_iteration": 0,
                "websites_to_load": WebsitesToLoad(websites=[]),
                "loaded_websites": [],
                "website_critiques": [],
                "selection": None,
            }
            for id in range(tries)
        ]

        responses: list[SearchAgentState] = self._workflow.batch(
            inputs, {"recursion_limit": 200}, return_exception=True
        )
        responses = [
            response for response in responses if not isinstance(response, Exception)
        ]

        logger.info(f"Aggregating results from {tries} runs.")

        aggregated_result = list(
            unique_everseen(
                (
                    website
                    for response in responses
                    for website in response["selection"].websites
                ),
                key=lambda choice: choice.website.link,
            )
        )

        return aggregated_result

    def _build_workflow(
        self,
    ) -> CompiledStateGraph[SearchAgentState, None, SearchAgentState, SearchAgentState]:
        """Builds and compiles the workflow.

        Returns:
            CompiledStateGraph[SearchAgentState, None, SearchAgentState, SearchAgentState]: execution-ready workflow.
        """
        workflow_graph = StateGraph(SearchAgentState)

        workflow_graph.add_node(SearchAgentNode.DESCRIPTION, self._description)
        workflow_graph.add_node(SearchAgentNode.SEARCH, self._search)
        workflow_graph.add_node(
            SearchAgentNode.TOOLS_SEARCHER, ToolNode(tools=[self._search_tool])
        )
        workflow_graph.add_node(SearchAgentNode.SELECT_PAGE, self._select_page)
        workflow_graph.add_node(SearchAgentNode.LOAD, self._load)
        workflow_graph.add_node(SearchAgentNode.CRITIQUE, self._critique)
        workflow_graph.add_node(SearchAgentNode.SUMMARY, self._summarize)

        workflow_graph.add_edge(START, SearchAgentNode.DESCRIPTION)
        workflow_graph.add_edge(SearchAgentNode.DESCRIPTION, SearchAgentNode.SEARCH)
        workflow_graph.add_edge(SearchAgentNode.SEARCH, SearchAgentNode.TOOLS_SEARCHER)
        workflow_graph.add_edge(
            SearchAgentNode.TOOLS_SEARCHER, SearchAgentNode.SELECT_PAGE
        )
        workflow_graph.add_edge(SearchAgentNode.SELECT_PAGE, SearchAgentNode.LOAD)
        workflow_graph.add_edge(SearchAgentNode.LOAD, SearchAgentNode.CRITIQUE)
        workflow_graph.add_conditional_edges(
            SearchAgentNode.CRITIQUE,
            self._decide_loop,
            {
                SearchAgentNode.SEARCH: SearchAgentNode.SEARCH,
                SearchAgentNode.SUMMARY: SearchAgentNode.SUMMARY,
            },
        )
        workflow_graph.add_edge(SearchAgentNode.SUMMARY, END)

        workflow = workflow_graph.compile()

        return workflow

    def _description(self, _: SearchAgentState) -> SearchAgentState:
        """Introduces the description as context.

        Returns:
            SearchAgentState: update to the state of the Agent
        """
        return {"messages": [SystemMessage(self._description_prompt)]}

    def _search(self, state: SearchAgentState) -> SearchAgentState:
        """Calls the search tool. Start of the search loop.

        Args:
            state (SearchAgentState): state of the Agent.

        Returns:
            SearchAgentState: update to the state of the Agent.
        """
        logger.info(f"run ID: {state['id']}. Searching for websites.")
        prompt = self._search_prompt

        response = self._model.bind_tools([self._search_tool]).invoke(
            state["messages"] + [HumanMessage(prompt)],
        )

        return {
            "messages": [HumanMessage(prompt), response],
            "search_loop_iteration": state["search_loop_iteration"] + 1,
        }

    def _select_page(self, state: SearchAgentState) -> SearchAgentState:
        """Selects websites to load from search results.

        Args:
            state (SearchAgentState): state of the Agent.

        Returns:
            SearchAgentState: update to the state of the Agent.
        """
        logger.info(f"run ID: {state['id']}. Selecting pages to visit.")

        prompt = self._select_page_prompt

        response = self._invoke_structured_model(
            WebsitesToLoad,
            state["messages"] + [HumanMessage(prompt)],
        )

        return {
            "messages": [
                HumanMessage(prompt),
                AIMessage(response.model_dump_json()),
            ],
            "websites_to_load": response,
        }

    def _load(self, state: SearchAgentState) -> SearchAgentState:
        """Loads websites' contents.

        Args:
            state (SearchAgentState): state of the Agent.

        Returns:
            SearchAgentState: update to the state of the Agent.
        """
        logger.info(f"run ID: {state['id']}. Loading websites.")

        results = [
            (website, self._load_website(website.link))
            for website in state["websites_to_load"].websites
        ]

        loaded_websites = [
            Website(header=website, content=content)
            for website, content in results
            if content
        ]

        return {
            "loaded_websites": state["loaded_websites"] + loaded_websites,
            "websites_to_load": [],
        }

    def _critique(self, state: SearchAgentState) -> SearchAgentState:
        """Calls the Critic for each loaded website.

        Args:
            state (SearchAgentState): state of the Agent.

        Returns:
            SearchAgentState: update to the state of the Agent.
        """
        logger.info(f"run ID: {state['id']}. Critiquing website candidates.")
        critiques = self._critic.run(state["loaded_websites"])

        return {
            "messages": [AIMessage(str(critiques))],
            "loaded_websites": [],
            "website_critiques": state["website_critiques"] + critiques,
        }

    def _summarize(self, state: SearchAgentState) -> SearchAgentState:
        """Calls the Selector to pick suitable websites and justify this decision.

        Args:
            state (SearchAgentState): state of the Agent.

        Returns:
            SearchAgentState: update to the state of the Agent.
        """
        logger.info(f"run ID: {state['id']}. Picking the best websites.")

        response = self._selector.run(state["website_critiques"])

        return {"selection": response}

    def _decide_loop(self, state: SearchAgentState) -> SearchAgentNode:
        """Decides whether to start a new search loop or return results.

        Returns:
            SearchAgentNode: next node to go to.
        """
        if state["search_loop_iteration"] == self._max_iterations:
            return SearchAgentNode.SUMMARY

        if state["search_loop_iteration"] < self._min_iterations:
            return SearchAgentNode.SEARCH

        prompt = self._decide_loop_prompt

        response = self._invoke_structured_model(
            LoopDecision,
            state["messages"] + [HumanMessage(prompt)],
        )

        return response.loop_decision

    @staticmethod
    def _load_website(url: str) -> str | None:
        """Loads website from the specified URL and returns its content.

        Args:
            url (str): url of the website.

        Returns:
            str | None: website content or None if failed to load.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        content = soup.get_text(separator=" ", strip=True)

        return content
