import logging

from dotenv import load_dotenv

import config
from tools import ddg_search
from web_crawler.crawler import Crawler

logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ddgs").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    """Example use of the Crawler"""
    load_dotenv()

    crawler = Crawler(
        search_tool=ddg_search,
        description_prompt=config.DESCRIPTION_PROMPT,
        search_search_prompt=config.SEARCH_SEARCH_PROMPT,
        search_select_page_prompt=config.SEARCH_SELECT_PAGE_PROMPT,
        search_decide_loop_prompt=config.SEARCH_DECIDE_LOOP_PROMPT,
        critic_introduction_prompt=config.CRITIC_INTRODUCTION_PROMPT,
        selector_introduction_prompt=config.SELECTOR_INTRODUCTION_PROMPT,
        iterations=config.ITERATIONS,
        model=config.MODEL,
        search_min_iterations=config.AGENT_MIN_ITERATIONS,
        search_max_iterations=config.AGENT_MAX_ITERATIONS,
    )

    found_websites = crawler.run()

    for website in found_websites:
        print(
            f"LINK: {website.website.link}\nJUSTIFICATION: {website.justification}\n\n"
        )


if __name__ == "__main__":
    main()
