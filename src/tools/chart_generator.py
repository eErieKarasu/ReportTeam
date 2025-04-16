import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Define output directory for charts
CHART_DIR = "output/charts"


def generate_bar_chart(data: List[Dict[str, Any]], category_col: str, value_col: str,
                       title: str = "Generated Bar Chart", xlabel: Optional[str] = None,
                       ylabel: Optional[str] = None) -> Optional[str]:
    """
    Generates a simple bar chart from list of dictionaries and saves it as a PNG file.

    Args:
        data: A list of dictionaries, e.g., [{'category': 'A', 'value': 10}, {'category': 'B', 'value': 20}].
        category_col: The key in the dictionaries to use for the category axis (X-axis).
        value_col: The key in the dictionaries to use for the value axis (Y-axis).
        title: The title of the chart.
        xlabel: Label for the X-axis. Defaults to category_col.
        ylabel: Label for the Y-axis. Defaults to value_col.

    Returns:
        The relative file path (from project root) of the saved chart PNG image,
        or None if chart generation fails.
    """
    logger.info(f"Attempting to generate bar chart: '{title}'")

    # Ensure output directory exists
    try:
        os.makedirs(CHART_DIR, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create chart directory {CHART_DIR}: {e}")
        return None

    # Validate input data structure (basic check)
    if not data or not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        logger.error("Invalid data format: Input must be a non-empty list of dictionaries.")
        return None
    if not category_col or not value_col:
        logger.error("Category column and value column names must be provided.")
        return None
    # Check if columns exist in the first data item (assuming consistent structure)
    if category_col not in data[0] or value_col not in data[0]:
        logger.error(f"Provided columns '{category_col}' or '{value_col}' not found in data.")
        return None
    # Check if value column is numeric (basic check on first item)
    first_value = data[0][value_col]
    if not isinstance(first_value, (int, float)):
        logger.warning(
            f"Value column '{value_col}' might not contain numeric data (first value: {first_value}). Chart might fail.")
        # Allow attempt, Matplotlib/Pandas might handle some cases or raise errors

    try:
        # Convert data to Pandas DataFrame for easier plotting
        df = pd.DataFrame(data)

        # Ensure value column is numeric if possible
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[value_col])  # Drop rows where conversion failed

        if df.empty:
            logger.error("DataFrame is empty after handling non-numeric values in value column.")
            return None

        # Create the plot
        plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        plt.bar(df[category_col], df[value_col])

        # Set labels and title
        plt.xlabel(xlabel if xlabel else category_col)
        plt.ylabel(ylabel if ylabel else value_col)
        plt.title(title)
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels if they overlap
        plt.tight_layout()  # Adjust layout

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize title for filename
        safe_title = "".join(c if c.isalnum() else "_" for c in title[:30]).strip('_')
        filename = f"chart_{timestamp}_{safe_title}.png"
        # Use os.path.join for cross-platform compatibility
        relative_filepath = os.path.join(CHART_DIR, filename)
        # Get absolute path for saving
        # save_path = os.path.abspath(relative_filepath) # Not needed if using relative path directly

        # Save the plot
        plt.savefig(relative_filepath)
        plt.close()  # Close the plot to free memory

        logger.info(f"Chart successfully generated and saved to: {relative_filepath}")
        # Return the relative path, which is needed for Markdown links
        return relative_filepath

    except Exception as e:
        logger.error(f"Failed to generate or save chart: {e}", exc_info=True)
        # Ensure plot is closed even if saving failed
        plt.close()
        return None


# Example Usage (optional, for testing the tool directly)
if __name__ == '__main__':
    from src.utils import setup_logging

    setup_logging(log_level=logging.DEBUG)

    # Sample data for testing
    sample_data = [
        {'Month': 'Jan', 'Sales': 150},
        {'Month': 'Feb', 'Sales': 200},
        {'Month': 'Mar', 'Sales': 120},
        {'Month': 'Apr', 'Sales': 250}
    ]

    print("\n--- Testing generate_bar_chart ---")
    chart_path = generate_bar_chart(
        data=sample_data,
        category_col='Month',
        value_col='Sales',
        title="Monthly Sales Data",
        xlabel="Month of Year",
        ylabel="Sales Amount ($)"
    )

    if chart_path:
        print(f"Test chart saved successfully to: {chart_path}")
        # You can manually check the output/charts directory
    else:
        print("Failed to generate test chart.")

    # Test with invalid data
    print("\n--- Testing with invalid data ---")
    invalid_data = [{'X': 'A', 'Y': 'ten'}, {'X': 'B', 'Y': 'twenty'}]
    chart_path_invalid = generate_bar_chart(invalid_data, 'X', 'Y', "Invalid Data Test")
    if not chart_path_invalid:
        print("Correctly failed to generate chart with invalid data.")
    else:
        print(f"Incorrectly generated chart with invalid data: {chart_path_invalid}")

