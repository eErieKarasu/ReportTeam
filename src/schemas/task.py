from dataclasses import dataclass

@dataclass
class Task:
    """
    Represents a task to be performed by an Agent.
    For the MVP, it simply contains the user's query.
    """
    query: str
    # Later, we might add fields like task_id, source_description, constraints, etc.
    