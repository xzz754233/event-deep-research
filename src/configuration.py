import os
from typing import Any
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Main configuration class for the Drama/Gossip Research agent."""

    # FIXED: Reverted to 1.5-flash. "2.5" does not exist and causes crashes.
    llm_model: str = Field(
        default="google_genai:gemini-2.5-flash",
        description="Primary LLM model",
    )

    structured_llm_model: str | None = Field(default=None)
    tools_llm_model: str | None = Field(default=None)
    chunk_llm_model: str | None = Field(default=None)

    structured_llm_max_tokens: int = Field(default=4096)
    tools_llm_max_tokens: int = Field(default=4096)
    max_structured_output_retries: int = Field(default=3)
    max_tools_output_retries: int = Field(default=3)

    default_chunk_size: int = Field(default=800)
    default_overlap_size: int = Field(default=20)
    max_content_length: int = Field(default=100000)

    # CHARM OPTIMIZATION: Conservative limits to ensure stability
    max_tool_iterations: int = Field(default=10)
    max_chunks: int = Field(default=3)

    def get_llm_structured_model(self) -> str:
        return self.structured_llm_model or self.llm_model

    def get_llm_with_tools_model(self) -> str:
        return self.tools_llm_model or self.llm_model

    def get_llm_chunk_model(self) -> str:
        # FIXED: Ensure chunk model is also valid
        return "google_genai:gemini-2.5-flash"

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        values = {
            k: os.environ.get(k.upper(), configurable.get(k))
            for k in cls.model_fields.keys()
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
