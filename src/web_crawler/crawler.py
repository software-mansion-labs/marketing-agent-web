import logging

from langchain.tools import BaseTool

from web_crawler.agents.search.agent import SearchAgent
from web_crawler.agents.critic.agent import CriticAgent
from web_crawler.agents.selector.agent import SelectorAgent
from web_crawler.agents.output_structures import WebsiteChoice


logger = logging.getLogger(__name__)


class Crawler:
    """Agentic Crawler for searching marketing opportunities."""

    def __init__(
        self,
        search_tool: BaseTool,
        description_prompt: str,
        search_search_prompt: str,
        search_select_page_prompt: str,
        search_decide_loop_prompt: str,
        critic_introduction_prompt: str,
        selector_introduction_prompt: str,
        iterations: int = 1,
        model: str = "openai:gpt-4o",
        search_loop_min_iterations: int = 2,
        search_loop_max_iterations: int = 5,
    ) -> None:
        """Initializes the agents to use.

        Args:
            search_tool (BaseTool): tool to use for searching on the web.
            description_prompt (str): prompt describing the product/providing general context.
            search_search_prompt (str): prompt telling the Search agent to search for specific websites.
            search_select_page_prompt (str): prompt telling the Search agent to select pages to load, providing criteria.
            search_decide_loop_prompt (str): prompt to decide whether to start a new iteration or summarize results.
            critic_introduction_prompt (str): prompt introducing the role of the Critic agent.
            selector_introduction_prompt (str): prompt introducing the role of the Selector agent.
            iterations (int, optional): number of times to run the entire agentic system. Defaults to 1.
            model (str, optional): foundation model. Defaults to "openai:gpt-4o".
            search_loop_min_iterations (int, optional): min number of iterations in a single run of the Search agent. Defaults to 2.
            search_loop_max_iterations (int, optional): max number of iterations in a single run of the Search agent. Defaults to 5.
        """

        critic = CriticAgent(
            introduction_prompt=critic_introduction_prompt,
            description_prompt=description_prompt,
            model=model,
        )
        selector = SelectorAgent(
            introduction_prompt=selector_introduction_prompt,
            description_prompt=description_prompt,
            model=model,
        )

        self._agent = SearchAgent(
            search_tool=search_tool,
            critic=critic,
            selector=selector,
            description_prompt=description_prompt,
            search_prompt=search_search_prompt,
            select_page_prompt=search_select_page_prompt,
            decide_loop_prompt=search_decide_loop_prompt,
            model=model,
            search_loop_min_iterations=search_loop_min_iterations,
            search_loop_max_iterations=search_loop_max_iterations,
        )
        self._iterations = iterations

    def run(self) -> list[WebsiteChoice]:
        """Runs the crawler and returns found websites.

        Returns:
            list[WebsiteChoice]: found websites.
        """

        result = self._agent.run(self._iterations)

        logger.info(f"All agents have completed their runs, found {len(result)} posts.")

        return result
