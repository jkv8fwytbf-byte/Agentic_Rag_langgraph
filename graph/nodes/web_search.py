from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")

    question = state["question"]
    # "documents" is not populated yet if the router sent us straight to web search
    documents: List[Document] = list(state.get("documents") or [])

    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        tavily_result["content"] for tavily_result in tavily_results
    )
    documents = documents + [Document(page_content=joined_tavily_result)]

    return {"documents": documents}
