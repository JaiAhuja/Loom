from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State schema for the Loom agent graph.

    Attributes:
        messages: Conversation message history. Uses add_messages reducer
                  to automatically append new messages.
    """

    messages: Annotated[list[BaseMessage], add_messages]
