from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime


@dataclass
class Report:
    """
    Represents the structured content of the generated report.
    """
    query: str  # The original user query
    plan: List[str]  # The list of section topics/titles used
    sections: Dict[str, str]  # Dictionary mapping section title to its generated content
    generation_timestamp: datetime = field(default_factory=datetime.now)  # Timestamp of generation

    # Optional: Add fields for metadata, sources, summary, etc. later
    # sources: Dict[str, List[str]] = field(default_factory=dict) # e.g., {'Section Title': ['url1', 'url2']}
    # overall_summary: Optional[str] = None

    def to_markdown(self) -> str:
        """Converts the report structure to a basic Markdown string."""
        md_string = f"# Report for Query: {self.query}\n\n"
        md_string += f"Generated on: {self.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md_string += "---\n\n"  # Separator

        for section_title in self.plan:  # Iterate in the planned order
            content = self.sections.get(section_title, "Error: Content not found for this section.")
            md_string += f"## {section_title}\n\n"  # Use H2 for sections
            md_string += f"{content}\n\n"  # Add section content
            md_string += "---\n\n"  # Separator between sections

        return md_string

    # You could add other export methods like to_json, to_html, etc.