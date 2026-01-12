# Marketing Agent Web :mag_right:

This package is a LangGraph multi-agent system searching the web for marketing opportunities for a product/company. 

## How to use it

First, you have to describe the thing you want to advertise in `src/config.py`, in `DESCRIPTION_PROMPT` constant. As an example, we provide a description of our product---[Private Mind](https://privatemind.swmansion.com/). You can also control the number of iterations agents run before aggregating the results, or prompts of specific sub-agents.

Once that's done, you create a `Crawler` object, as shown in `src/main.py`. The crawler is all set and you can run the search. It returns a list of websites suitable for advertisement, along with justifications of its picks.

### Web-Search Tool

`Crawler` takes in a search tool as one of its arguments. This is the tool that the agents use to search the web. It's customizable, and we provide an example tool in `src/tools`, based on the DuckDuckGo search engine. If you decide to use a search engine that requires a key, specify it in `.env`. 

### Models

By default, OpenAI models are available through `langchain-openai` dependency. Other models are also supported, but you need to install their packages to use them (see the [integrations page](https://docs.langchain.com/oss/python/integrations/providers/overview)). Once you've installed a specific package, you just change the name of the model in `src/config.py` accordingly and provide a key in `.env`.