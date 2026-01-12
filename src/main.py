import logging

from dotenv import load_dotenv

import config
from tools.ddg_search import ddg_search
from web_crawler.crawler import Crawler

load_dotenv()

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
        search_loop_min_iterations=config.SEARCH_SEARCH_LOOP_MIN_ITERATIONS,
        search_loop_max_iterations=config.SEARCH_SEARCH_LOOP_MAX_ITERATIONS,
    )

    found_websites = crawler.run()

    for website in found_websites:
        print(website)


if __name__ == "__main__":
    main()
