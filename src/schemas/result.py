from dataclasses import dataclass
from typing import Any # Using Any for now, can be refined later

@dataclass
class Result:
    """
    Represents the result of an Agent's task execution.
    For the MVP, it primarily contains the processed content.
    """
    content: Any # Could be raw text, list of snippets, structured data, etc.
    source_agent: str # Name or type of the agent that produced this result
    # Later, we might add fields like status, metadata, error_info, etc.
