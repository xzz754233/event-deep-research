import json
from typing import List, Literal, TypedDict

from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.state import Command, RunnableConfig

# FIXED: Use standard asyncio, NOT langgraph's internal one
import asyncio
from pydantic import BaseModel, Field
from src.configuration import Configuration
from src.llm_service import create_llm_with_tools

from src.research_events.chunk_graph import create_drama_event_graph
from src.research_events.merge_events.prompts import (
    EXTRACT_AND_CATEGORIZE_PROMPT,
    MERGE_EVENTS_TEMPLATE,
)
from src.research_events.merge_events.utils import ensure_categories_with_events
from src.services.event_service import EventService
from src.state import CategoriesWithEvents
from src.url_crawler.utils import chunk_text_by_tokens
from src.utils import get_langfuse_handler


class RelevantEventsCategorized(BaseModel):
    """The chunk contains relevant drama/scandal events that have been categorized."""

    context: str = Field(description="Background info, previous relationships, origin.")
    conflict: str = Field(description="The main incident, accusations, leaks.")
    reaction: str = Field(description="Public responses, tweets, lawsuits.")
    outcome: str = Field(description="Current status, cancellations, resolution.")


class IrrelevantChunk(BaseModel):
    """The chunk contains NO drama/scandal events."""


class InputMergeEventsState(TypedDict):
    existing_events: CategoriesWithEvents
    extracted_events: str
    research_question: str


class MergeEventsState(InputMergeEventsState):
    text_chunks: List[str]
    categorized_chunks: List[CategoriesWithEvents]
    extracted_events_categorized: CategoriesWithEvents


class OutputMergeEventsState(TypedDict):
    existing_events: CategoriesWithEvents


async def split_events(
    state: MergeEventsState,
) -> Command[Literal["filter_chunks", "__end__"]]:
    """Use token-based chunking."""
    extracted_events = state.get("extracted_events", "")
    if not extracted_events.strip():
        return Command(
            goto="__end__", update={"text_chunks": [], "categorized_chunks": []}
        )

    chunks = await chunk_text_by_tokens(extracted_events)
    return Command(
        goto="filter_chunks",
        update={"text_chunks": chunks[0:20], "categorized_chunks": []},
    )


async def filter_chunks(
    state: MergeEventsState, config: RunnableConfig
) -> Command[Literal["extract_and_categorize_chunk", "__end__"]]:
    """Filter chunks using the drama detection graph."""
    chunks = state.get("text_chunks", [])
    if not chunks:
        return Command(goto="__end__")

    chunk_graph = create_drama_event_graph()
    configurable = Configuration.from_runnable_config(config)

    # Slice chunks to max limit
    processing_chunks = (
        chunks[: configurable.max_chunks]
        if len(chunks) > configurable.max_chunks
        else chunks
    )

    relevant_chunks = []
    # Process sequentially to avoid event loop overload
    for chunk in processing_chunks:
        try:
            chunk_result = await chunk_graph.ainvoke({"text": chunk}, config)
            has_events = any(
                result.contains_drama_event
                for result in chunk_result["results"].values()
            )
            if has_events:
                relevant_chunks.append(chunk)
        except Exception as e:
            print(f"Error filtering chunk: {e}")

    if not relevant_chunks:
        return Command(goto="__end__")

    return Command(
        goto="extract_and_categorize_chunk",
        update={"text_chunks": relevant_chunks, "categorized_chunks": []},
    )


async def extract_and_categorize_chunk(
    state: MergeEventsState, config: RunnableConfig
) -> Command[Literal["extract_and_categorize_chunk", "merge_categorizations"]]:
    """Extract and categorize events from a chunk."""
    chunks = state.get("text_chunks", [])
    categorized_chunks = state.get("categorized_chunks", [])

    if len(categorized_chunks) >= len(chunks):
        return Command(goto="merge_categorizations")

    chunk = chunks[len(categorized_chunks)]
    prompt = EXTRACT_AND_CATEGORIZE_PROMPT.format(text_chunk=chunk)

    tools = [tool(RelevantEventsCategorized), tool(IrrelevantChunk)]
    model = create_llm_with_tools(tools=tools, config=config)

    try:
        response = await model.ainvoke(prompt)

        # DEFENSIVE CODING: Check if tool calls exist
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = tool_call["args"]

            # GEMINI FIX: Parse JSON string if needed
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}

            if tool_call["name"] == "RelevantEventsCategorized":
                # Convert list values to strings if necessary
                clean_args = {
                    k: ("\n".join(v) if isinstance(v, list) else v)
                    for k, v in args.items()
                }
                categorized = CategoriesWithEvents(**clean_args)
            else:
                categorized = CategoriesWithEvents(
                    context="", conflict="", reaction="", outcome=""
                )
        else:
            categorized = CategoriesWithEvents(
                context="", conflict="", reaction="", outcome=""
            )

    except Exception as e:
        print(f"Error categorizing chunk: {e}")
        categorized = CategoriesWithEvents(
            context="", conflict="", reaction="", outcome=""
        )

    return Command(
        goto="extract_and_categorize_chunk",
        update={"categorized_chunks": categorized_chunks + [categorized]},
    )


async def merge_categorizations(
    state: MergeEventsState,
) -> Command[Literal["combine_new_and_original_events"]]:
    """Merge all results."""
    results = state.get("categorized_chunks", [])
    merged = EventService.merge_categorized_events(results)
    return Command(
        goto="combine_new_and_original_events",
        update={"extracted_events_categorized": merged},
    )


async def combine_new_and_original_events(
    state: MergeEventsState, config: RunnableConfig
) -> Command:
    """Combine with LLM."""
    print("Combining new and original events...")

    existing_events_raw = state.get("existing_events")
    new_events_raw = state.get("extracted_events_categorized")

    existing_events = ensure_categories_with_events(existing_events_raw)
    new_events = ensure_categories_with_events(new_events_raw)

    if not new_events or not any(
        getattr(new_events, cat, "").strip()
        for cat in CategoriesWithEvents.model_fields.keys()
    ):
        return Command(goto="__end__", update={"existing_events": existing_events})

    merge_tasks = []
    categories = CategoriesWithEvents.model_fields.keys()

    # Use regular structured model
    from src.llm_service import create_llm_structured_model

    llm = create_llm_structured_model(config=config)

    for category in categories:
        existing_text = getattr(existing_events, category, "").strip()
        new_text = getattr(new_events, category, "").strip()

        if not (existing_text or new_text):
            continue

        existing_display = existing_text if existing_text else "No events"
        new_display = new_text if new_text else "No events"

        prompt = MERGE_EVENTS_TEMPLATE.format(
            original=existing_display, new=new_display
        )
        merge_tasks.append((category, llm.ainvoke(prompt)))

    final_merged_dict = {}
    if merge_tasks:
        cats, tasks = zip(*merge_tasks)
        responses = await asyncio.gather(*tasks)
        final_merged_dict = {cat: resp.content for cat, resp in zip(cats, responses)}

    for category in CategoriesWithEvents.model_fields.keys():
        if category not in final_merged_dict:
            final_merged_dict[category] = getattr(existing_events, category, "")

    final_merged_output = CategoriesWithEvents(**final_merged_dict)
    return Command(goto="__end__", update={"existing_events": final_merged_output})


merge_events_graph_builder = StateGraph(
    MergeEventsState, input_schema=InputMergeEventsState, config_schema=Configuration
)
merge_events_graph_builder.add_node("split_events", split_events)
merge_events_graph_builder.add_node("filter_chunks", filter_chunks)
merge_events_graph_builder.add_node(
    "extract_and_categorize_chunk", extract_and_categorize_chunk
)
merge_events_graph_builder.add_node("merge_categorizations", merge_categorizations)
merge_events_graph_builder.add_node(
    "combine_new_and_original_events", combine_new_and_original_events
)
merge_events_graph_builder.add_edge(START, "split_events")

merge_events_app = merge_events_graph_builder.compile().with_config(
    {"callbacks": [get_langfuse_handler()], "recursionLimit": 200}
)
