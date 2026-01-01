import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel, Field

################################################################################
# Section 1: Core Data Models
# - Defines the structure of the primary research output: the chronological timeline.
################################################################################


class ChronologyDate(BaseModel):
    """A structured representation of a date for a chronological event."""

    year: int | None = Field(None, description="The year of the event.")
    note: str | None = Field(
        None,
        description="Adds extra information to the date (month, day, specific time, e.g., 'May 5th, 3:00 PM').",
    )


class ChronologyEventInput(BaseModel):
    """Represents a single event, typically used for initial data extraction before an ID is assigned."""

    name: str = Field(description="A short, catchy title...")
    description: str = Field(description="A concise description...")
    date: ChronologyDate = Field(..., description="The structured date...")
    location: str = Field(
        default="",
        description="The platform (Twitter/X) or physical location. If unknown, leave empty.",
    )


class ChronologyEvent(ChronologyEventInput):
    """The final, canonical event model with a unique identifier."""

    id: str = Field(
        description="The id of the event in lowercase and underscores. Ex: 'drama_incident_1'"
    )


class ChronologyInput(BaseModel):
    """A list of newly extracted events from a research source."""

    events: list[ChronologyEventInput]


class Chronology(BaseModel):
    """A complete chronological timeline with finalized (ID'd) events."""

    events: list[ChronologyEvent]


class CategoriesWithEvents(BaseModel):
    # CHANGED: Renamed from 'early' to 'context'
    context: str = Field(
        default="",
        description="Background info, previous relationships, or the 'calm before the storm'. origin of the beef.",
    )
    # CHANGED: Renamed from 'personal' to 'conflict'
    conflict: str = Field(
        default="",
        description="The main incident, the accusation, the leak, the breakup, or the scandal itself.",
    )
    # CHANGED: Renamed from 'career' to 'reaction'
    reaction: str = Field(
        default="",
        description="Public responses, PR statements, tweets from other influencers, lawsuits, or 'receipts' posted.",
    )
    # CHANGED: Renamed from 'legacy' to 'outcome'
    outcome: str = Field(
        default="",
        description="Current status, who was cancelled, impact on career, or final resolution (if any).",
    )


################################################################################
# Section 2: Agent Tools
# - Pydantic models that define the tools available to the LLM agents.
################################################################################


class ResearchEventsTool(BaseModel):
    """The query to be used to research the drama/scandal. The query is based on the reflection of the assistant."""

    research_question: str
    pass  # No arguments needed


class FinishResearchTool(BaseModel):
    """Concludes the research process.
    Call this tool ONLY when you have a full picture of the drama, covering the context,
    the main conflict, key reactions, and the current outcome.
    """

    pass


################################################################################
# Section 3: Graph State Definitions
# - TypedDicts and models that define the "memory" for the agent graphs.
################################################################################


def override_reducer(current_value, new_value):
    """Reducer function that allows a new value to completely replace the old one."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


# --- Main Supervisor Graph State ---


class SupervisorStateInput(TypedDict):
    """The initial input to start the main research graph."""

    person_to_research: str  # Can be a person name or a topic (e.g. "FTX Collapse")
    existing_events: CategoriesWithEvents = Field(
        # CHANGED: Updated default to use new category names
        default=CategoriesWithEvents(context="", conflict="", reaction="", outcome=""),
        description="Covers chronology events of the topic.",
    )
    used_domains: list[str] = Field(
        default=[],
        description="The domains that have been used to extract events.",
    )
    events_summary: str = Field(
        default="",
        description="A summary of the events.",
    )


class SupervisorState(SupervisorStateInput):
    """The complete state for the main supervisor graph."""

    final_events: List[ChronologyEvent]
    conversation_history: Annotated[list[MessageLikeRepresentation], override_reducer]
    iteration_count: int = 0
    structured_events: list[ChronologyEvent] | None
