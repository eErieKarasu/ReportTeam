import logging
from typing import Optional, List, Dict, Any
from src.schemas import Task, Result
from src.tools import generate_bar_chart  # Import the chart generation tool

logger = logging.getLogger(__name__)


class ChartGenerationAgent:
    """
    An agent responsible for generating charts from structured data
    extracted by the AnalysisAgent.
    """

    def __init__(self):
        # Potential future config: preferred chart types, libraries, etc.
        pass

    def _find_chartable_columns(self, data: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """
        Simple heuristic to find the first suitable category (string) column
        and the first suitable value (numeric) column for a bar chart.

        Args:
            data: A list of dictionaries (structured data).

        Returns:
            A dictionary {'category_col': name, 'value_col': name} if suitable
            columns are found, otherwise None.
        """
        if not data:
            return None

        first_item = data[0]
        category_col = None
        value_col = None

        # Find first string-like column for category
        for key, value in first_item.items():
            if isinstance(value, str):
                category_col = key
                break  # Use the first one found

        # Find first numeric column for value (starting after category potentially)
        for key, value in first_item.items():
            if isinstance(value, (int, float)):
                # Avoid using the same column if possible, unless it's the only numeric one
                if key != category_col or value_col is None:
                    value_col = key
                    # Prefer not using the category col as value if another numeric exists
                    if key != category_col:
                        break

        if category_col and value_col:
            logger.debug(f"Found potential chart columns: Category='{category_col}', Value='{value_col}'")
            return {'category_col': category_col, 'value_col': value_col}
        else:
            logger.warning(
                f"Could not automatically determine suitable category/value columns from data keys: {list(first_item.keys())}")
            return None

    def run(self, analysis_result: Result, section_topic: str) -> Optional[Result]:
        """
        Attempts to generate a chart based on the structured data in the analysis result.

        Args:
            analysis_result: The Result object from AnalysisAgent, containing structured data.
            section_topic: The topic of the current section (used for chart title).

        Returns:
            A Result object where content is the file path of the generated chart,
            or None if no suitable data was found or chart generation failed.
        """
        logger.info(f"ChartGenerationAgent received analysis result for section: '{section_topic}'")

        if not analysis_result or not isinstance(analysis_result.content, dict):
            logger.warning("ChartGenerationAgent received invalid analysis result format.")
            return None

        analysis_data = analysis_result.content
        structured_data = analysis_data.get('structured_data')

        if not structured_data or not isinstance(structured_data, list) or len(structured_data) == 0:
            logger.info(f"No suitable structured data found for section '{section_topic}'. Skipping chart generation.")
            return None

        logger.info(f"Found structured data for section '{section_topic}'. Attempting chart generation.")
        logger.debug(f"Structured data sample: {structured_data[:2]}")  # Log first few items

        # --- Determine chart parameters ---
        # Use heuristic to find columns
        column_info = self._find_chartable_columns(structured_data)

        if not column_info:
            logger.error(
                f"Could not determine columns for charting in section '{section_topic}'. Skipping chart generation.")
            return None

        category_col = column_info['category_col']
        value_col = column_info['value_col']
        # Generate a title based on the section topic and columns
        chart_title = f"{section_topic}: {value_col} by {category_col}"

        # --- Call the chart generation tool ---
        chart_filepath = generate_bar_chart(
            data=structured_data,
            category_col=category_col,
            value_col=value_col,
            title=chart_title,
            # Labels can be inferred or explicitly set
            xlabel=category_col,
            ylabel=value_col
        )

        if chart_filepath:
            logger.info(f"Chart successfully generated for section '{section_topic}' at: {chart_filepath}")
            # Return the file path as the result content
            return Result(content=chart_filepath, source_agent="ChartGenerationAgent")
        else:
            logger.error(f"Chart generation failed for section '{section_topic}'.")
            return None


# Example usage (optional, for testing the agent directly)
if __name__ == '__main__':
    from src.utils import setup_logging

    setup_logging(log_level=logging.DEBUG)

    # Mock Analysis Result with structured data
    mock_analysis_content = {
        'sufficient': True,
        'key_points': ['Point 1', 'Point 2'],
        'structured_data': [
            {'Brand': 'Apple', 'MarketShare': 25.5},
            {'Brand': 'Samsung', 'MarketShare': 20.1},
            {'Brand': 'Xiaomi', 'MarketShare': 15.3},
            {'Brand': 'Oppo', 'MarketShare': 10.0}
        ],
        'suggestion_for_further_research': None
    }
    mock_analysis_result = Result(content=mock_analysis_content, source_agent="MockAnalysisAgent")
    test_section = "Smartphone Market Share Q1"

    chart_agent = ChartGenerationAgent()
    chart_result = chart_agent.run(analysis_result=mock_analysis_result, section_topic=test_section)

    if chart_result:
        print(f"\n--- ChartGenerationAgent Result ---")
        print(f"Source: {chart_result.source_agent}")
        print(f"Generated Chart Path: {chart_result.content}")
        # Check output/charts directory for the generated file
    else:
        print("\nChartGenerationAgent did not produce a chart.")

    # Test case with unsuitable data
    mock_analysis_content_bad_data = {
        'sufficient': True, 'key_points': ['...'],
        'structured_data': [{'Text': 'abc', 'Description': 'def'}],  # No numeric data
        'suggestion_for_further_research': None
    }
    mock_analysis_result_bad = Result(content=mock_analysis_content_bad_data, source_agent="MockAnalysisAgent")
    test_section_bad = "Bad Data Test"
    print("\n--- Testing with unsuitable data ---")
    chart_result_bad = chart_agent.run(analysis_result=mock_analysis_result_bad, section_topic=test_section_bad)
    if not chart_result_bad:
        print("Correctly skipped chart generation for unsuitable data.")
    else:
        print(f"Incorrectly generated chart for unsuitable data: {chart_result_bad.content}")
