import os
from typing import Any
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Main configuration class for the Drama/Gossip Research agent."""

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

    # 限制 20k 字元，避免 40萬字網頁卡死
    max_content_length: int = Field(default=20000)

    # 恢復到正常的 5 次，給它足夠空間思考
    max_tool_iterations: int = Field(default=5)
    max_chunks: int = Field(default=3)

    def get_llm_structured_model(self) -> str:
        return self.structured_llm_model or self.llm_model

    def get_llm_with_tools_model(self) -> str:
        return self.tools_llm_model or self.llm_model

    def get_llm_chunk_model(self) -> str:
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
