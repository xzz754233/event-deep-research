import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from src.configuration import Configuration
from src.llm_service import create_llm_structured_model, create_llm_with_tools
from src.prompts import (
    events_summarizer_prompt,
    lead_researcher_prompt,
    structure_events_prompt,
)
from src.research_events.research_events_graph import research_events_app

# IMPORT FIX: Need this helper to handle Dict/Object ambiguity
from src.research_events.merge_events.utils import ensure_categories_with_events
from src.state import (
    CategoriesWithEvents,
    Chronology,
    FinishResearchTool,
    ResearchEventsTool,
    SupervisorState,
    SupervisorStateInput,
)
from src.utils import get_buffer_string_with_tools, get_langfuse_handler, think_tool

config = Configuration()
MAX_TOOL_CALL_ITERATIONS = config.max_tool_iterations


async def supervisor_node(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor_tools"]]:
    """The 'brain' of the agent."""
    tools = [ResearchEventsTool, FinishResearchTool, think_tool]
    tools_model = create_llm_with_tools(tools=tools, config=config)

    messages = state.get("conversation_history", [])
    # Safe fallback if history is empty
    last_message = messages[-1] if messages else ""

    system_message = SystemMessage(
        content=lead_researcher_prompt.format(
            person_to_research=state["person_to_research"],
            events_summary=state.get("events_summary", "Everything is missing"),
            last_message=last_message,
            max_iterations=MAX_TOOL_CALL_ITERATIONS,
        )
    )

    human_message = HumanMessage(content="Start the research process.")
    prompt = [system_message, human_message]
    response = await tools_model.ainvoke(prompt)

    return Command(
        goto="supervisor_tools",
        update={
            "conversation_history": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        },
    )


async def supervisor_tools_node(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor", "structure_events"]]:
    """The 'hands' of the agent. Executes tools."""

    # DEFENSIVE CODING: Normalize state data
    raw_events = state.get("existing_events")
    # If None, create default. If Dict or Object, normalize to Object.
    if not raw_events:
        existing_events = CategoriesWithEvents(
            context="", conflict="", reaction="", outcome=""
        )
    else:
        existing_events = ensure_categories_with_events(raw_events)

    events_summary = state.get("events_summary", "")
    used_domains = state.get("used_domains", [])
    last_message = state["conversation_history"][-1]
    iteration_count = state.get("iteration_count", 0)

    if not last_message.tool_calls or iteration_count >= MAX_TOOL_CALL_ITERATIONS:
        return Command(goto="structure_events")

    all_tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # GEMINI FIX: Parse JSON string args if necessary
        if isinstance(tool_args, str):
            print(f"⚠️ Tool args is string, parsing: {tool_args[:50]}...")
            try:
                tool_args = json.loads(tool_args)
            except Exception as e:
                print(f"❌ JSON Parse Error: {e}")
                tool_args = {}

        if tool_name == "FinishResearchTool":
            return Command(goto="structure_events")

        elif tool_name == "think_tool":
            response_content = tool_args.get("reflection", "Reflection recorded.")
            all_tool_messages.append(
                ToolMessage(
                    content=response_content,
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

        elif tool_name == "ResearchEventsTool":
            research_question = tool_args.get("research_question", "")

            # Call Sub-graph
            # Note: We pass the Pydantic object, LangGraph might serialize it back to dict
            result = await research_events_app.ainvoke(
                {
                    "research_question": research_question,
                    "existing_events": existing_events,
                    "used_domains": used_domains,
                }
            )

            # Update local variables from result
            # Ensure normalization again just in case
            existing_events = ensure_categories_with_events(result["existing_events"])
            used_domains = result["used_domains"]

            # Summarize
            summarizer_prompt = events_summarizer_prompt.format(
                existing_events=existing_events
            )
            response = await create_llm_structured_model(config=config).ainvoke(
                summarizer_prompt
            )
            events_summary = response.content

            all_tool_messages.append(
                ToolMessage(
                    content=f"Research complete for: {research_question}",
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

    return Command(
        goto="supervisor",
        update={
            "existing_events": existing_events,
            "conversation_history": all_tool_messages,
            "used_domains": used_domains,
            "events_summary": events_summary,
        },
    )


async def structure_events(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """Step 2: Structures the events into JSON."""
    print("--- Step 2: Structuring Events into JSON ---")

    # DEFENSIVE CODING: Normalize state data
    raw_events = state.get("existing_events")
    if not raw_events:
        print("Warning: No events found.")
        return {"structured_events": []}

    # Crucial: Ensure it's a Pydantic object so we can use dot notation
    existing_events = ensure_categories_with_events(raw_events)

    structured_llm = create_llm_structured_model(config=config, class_name=Chronology)

    # Now dot notation is safe
    # Use specific prompts for each category, appending the category text to help context
    tasks = [
        (
            structure_events_prompt.format(existing_events=existing_events.context),
            "context",
        ),
        (
            structure_events_prompt.format(existing_events=existing_events.conflict),
            "conflict",
        ),
        (
            structure_events_prompt.format(existing_events=existing_events.reaction),
            "reaction",
        ),
        (
            structure_events_prompt.format(existing_events=existing_events.outcome),
            "outcome",
        ),
    ]

    # Run sequentially to ensure stability
    all_events = []
    for prompt, category in tasks:
        try:
            resp = await structured_llm.ainvoke(prompt)
            # Add to master list
            all_events.extend(resp.events)
        except Exception as e:
            print(f"Error structuring category {category}: {e}")

    # --- 🛡️ GOLDEN FIX: POST-PROCESSING CLEANUP ---
    # This block fixes the "\" cut-off issue and removes "Unknown" dates programmatically.
    cleaned_events = []
    for event in all_events:
        # 1. Aggressive Strip: Removes spaces, backslashes, and accidental quotes from ends
        if event.description:
            # Strip whitespace, then strip backslashes, then strip quotes
            event.description = event.description.strip().strip("\\").strip('"').strip()

        if event.name:
            event.name = event.name.strip().strip("\\").strip('"').strip()

        # 2. Fix "Unknown" Locations/Dates
        if not event.location or event.location.lower() in ["none", "unknown", "null"]:
            event.location = "Internet / General"

        # 3. Last Line Defense: Add period if missing (and not empty)
        if event.description and not event.description.endswith((".", "!", "?", '"')):
            event.description += "."

        cleaned_events.append(event)
    # --------------------------------------------------

    return {
        "structured_events": cleaned_events,
    }


workflow = StateGraph(SupervisorState, input_schema=SupervisorStateInput)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("supervisor_tools", supervisor_tools_node)
workflow.add_node("structure_events", structure_events)
workflow.add_edge(START, "supervisor")

graph = workflow.compile().with_config({"callbacks": [get_langfuse_handler()]})
