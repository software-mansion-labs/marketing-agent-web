from typing import Any

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults


@tool(parse_docstring=True)
def ddg_search(query: str, num_results: int = 10) -> Any:
    """Runs the search with DuckDuckGo search engine and returns results.

    Args:
        query (str): query to search.
        num_results (int): number of results to return. Defaults to 10.

    Returns:
        Any: found websites.
    """
    search = DuckDuckGoSearchResults(output_format="list", num_results=num_results)
    return search.invoke(query)
