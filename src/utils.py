import os
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


@tool(
    description="Mandatory reflection tool. Analyze results and plan the next search query."
)
def think_tool(reflection: str) -> str:
    """Mandatory reflection step."""
    return f"Reflection recorded. {reflection}"


def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for a specific model from environment or config."""
    model_name = model_name.lower()
    if model_name.startswith("openai:"):
        return os.getenv("OPENAI_API_KEY")
    elif model_name.startswith("anthropic:"):
        return os.getenv("ANTHROPIC_API_KEY")
    elif model_name.startswith("google"):
        # SECURITY FIX: Removed print statement exposing API Key
        return os.getenv("GOOGLE_API_KEY")
    return None


def get_buffer_string_with_tools(messages: list[BaseMessage]) -> str:
    """Return a readable transcript showing roles."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Human: {m.content}")
        elif isinstance(m, AIMessage):
            ai_content = f"AI: {m.content}"
            if hasattr(m, "tool_calls") and m.tool_calls:
                tool_calls_str = ", ".join(
                    [
                        f"{tc.get('name', 'unknown')}({tc.get('args', {})})"
                        for tc in m.tool_calls
                    ]
                )
                ai_content += f" [Tool calls: {tool_calls_str}]"
            lines.append(ai_content)
        elif isinstance(m, SystemMessage):
            lines.append(f"System: {m.content}")
        elif isinstance(m, ToolMessage):
            tool_name = (
                getattr(m, "name", None) or getattr(m, "tool", None) or "unknown_tool"
            )
            lines.append(f"Tool[{tool_name}]: {m.content}")
        else:
            lines.append(f"{m.__class__.__name__}: {m.content}")
    return "\n".join(lines)


def get_langfuse_handler():
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except ImportError:
        return None
