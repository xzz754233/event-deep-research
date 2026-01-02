from typing import Literal, TypedDict

from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field
from src.configuration import Configuration
from src.llm_service import create_llm_structured_model
from src.research_events.merge_events.merge_events_graph import merge_events_app
from src.services.url_service import URLService
from src.state import CategoriesWithEvents
from src.url_crawler.url_krawler_graph import url_crawler_app
from src.utils import get_langfuse_handler

# IMPORT FIX: Ensure we use the helper
from src.research_events.merge_events.utils import ensure_categories_with_events


class InputResearchEventsState(TypedDict):
    research_question: str
    existing_events: CategoriesWithEvents
    used_domains: list[str]


class ResearchEventsState(InputResearchEventsState):
    urls: list[str]
    extracted_events: str


class OutputResearchEventsState(TypedDict):
    existing_events: CategoriesWithEvents
    used_domains: list[str]


class BestUrls(BaseModel):
    selected_urls: list[str] = Field(
        description="A list containing ONLY the single best URL."
    )


def url_finder(
    state: ResearchEventsState,
    config: RunnableConfig,
) -> Command[Literal["should_process_url_router"]]:
    """Find the urls for the research_question"""
    research_question = state.get("research_question", "")
    used_domains = state.get("used_domains", [])

    if not research_question:
        raise ValueError("research_question is required")

    tool = TavilySearch(
        max_results=3,
        topic="general",
        include_raw_content=False,
        include_answer=False,
        exclude_domains=used_domains,
    )

    result = tool.invoke({"query": research_question})
    urls = [result["url"] for result in result["results"]]

    prompt = """
        From the results below, select the ONE SINGLE best URL that provides a comprehensive timeline 
        of the drama/scandal. Prefer Wikipedia, BBC, or major news recaps.

        <Results>
        {results}
        </Results>

        <Research Question>
        {research_question}
        </Research Question>
    """

    prompt = prompt.format(results=urls, research_question=research_question)
    structured_llm = create_llm_structured_model(config=config, class_name=BestUrls)
    structured_result = structured_llm.invoke(prompt)

    return Command(
        goto="should_process_url_router",
        update={"urls": structured_result.selected_urls},
    )


def updateUrlList(
    state: ResearchEventsState,
) -> tuple[list[str], list[str]]:
    urls = state.get("urls", [])
    used_domains = state.get("used_domains", [])
    return URLService.update_url_list(urls, used_domains)


def should_process_url_router(
    state: ResearchEventsState,
) -> Command[Literal["crawl_url", "__end__"]]:
    urls = state.get("urls", [])
    used_domains = state.get("used_domains", [])

    if urls and len(urls) > 0:
        domain = URLService.extract_domain(urls[0])
        if domain in used_domains:
            remaining_urls = urls[1:]
            return Command(
                goto="should_process_url_router",
                update={"urls": remaining_urls, "used_domains": used_domains},
            )

        print(f"URLs remaining: {len(state['urls'])}. Routing to crawl.")
        return Command(goto="crawl_url")
    else:
        print("No URLs remaining. Routing to __end__.")
        return Command(goto=END)


async def crawl_url(
    state: ResearchEventsState,
) -> Command[Literal["merge_events_and_update"]]:
    """Crawls the next URL."""
    urls = state["urls"]
    url_to_process = urls[0]
    research_question = state.get("research_question", "")

    if not research_question:
        raise ValueError("research_question is required for url crawling")

    result = await url_crawler_app.ainvoke(
        {"url": url_to_process, "research_question": research_question}
    )
    extracted_events = result["extracted_events"]

    return Command(
        goto="merge_events_and_update",
        update={"extracted_events": extracted_events},
    )


async def merge_events_and_update(
    state: ResearchEventsState,
) -> Command[Literal["should_process_url_router"]]:
    """Merges new events."""
    # Normalize state
    existing_events_raw = state.get("existing_events")
    existing_events = ensure_categories_with_events(existing_events_raw)

    extracted_events = state.get("extracted_events", "")
    research_question = state.get("research_question", "")

    try:
        # Invoke the merge subgraph
        result = await merge_events_app.ainvoke(
            {
                "existing_events": existing_events,
                "extracted_events": extracted_events,
                "research_question": research_question,
            }
        )

        # DEFENSIVE CODING: Check output type
        if isinstance(result, dict) and "existing_events" in result:
            new_existing_events = result["existing_events"]
        else:
            print(
                f"⚠️ Merge subgraph returned unexpected format: {type(result)}. Retaining old events."
            )
            new_existing_events = existing_events

    except Exception as e:
        print(f"❌ Error in merge_events_app: {e}. Retaining old events.")
        new_existing_events = existing_events

    remaining_urls, used_domains = updateUrlList(state)

    return Command(
        goto="should_process_url_router",
        update={
            "existing_events": new_existing_events,
            "urls": remaining_urls,
            "used_domains": used_domains,
        },
    )


research_events_builder = StateGraph(
    ResearchEventsState,
    input_schema=InputResearchEventsState,
    output_schema=OutputResearchEventsState,
    config_schema=Configuration,
)

research_events_builder.add_node("url_finder", url_finder)
research_events_builder.add_node("should_process_url_router", should_process_url_router)
research_events_builder.add_node("crawl_url", crawl_url)
research_events_builder.add_node("merge_events_and_update", merge_events_and_update)

research_events_builder.add_edge(START, "url_finder")

research_events_app = research_events_builder.compile().with_config(
    {"callbacks": [get_langfuse_handler()]}
)
