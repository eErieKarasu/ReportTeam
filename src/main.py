import logging
import asyncio
from src.agents import PlannerAgent, ResearchAgent, AnalysisAgent, WriterAgent, ChartGenerationAgent
from src.schemas import Task, Report
from src.core import Orchestrator
from src.utils import setup_logging
import sys
import os
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

# --- NEW: Define output directory ---
REPORT_DIR = "output/reports"

# --- MODIFIED: Changed to async def ---
async def run_report_generation(query: str):
    """
    Asynchronously runs the full report generation pipeline including chart generation.
    """
    # Logging setup can remain sync
    setup_logging(log_level=logging.DEBUG) # Keep DEBUG for testing charts
    logger.info("--- Starting Full Report Generation Pipeline ---")
    logger.info(f"Original Query: {query}")

    # 1. Create the initial task
    initial_task = Task(query=query)

    # --- NEW: Instantiate Agents ---
    # We need instances of all agents to potentially pass to the orchestrator
    # or use directly (like Planner).
    try:
        planner = PlannerAgent()
        # Configure research agent (e.g., how many pages to fetch)
        researcher = ResearchAgent(max_pages_to_fetch=3) # Adjust as needed
        analyzer = AnalysisAgent()
        writer = WriterAgent()
        # --- NEW: Instantiate ChartGenerationAgent ---
        chart_generator = ChartGenerationAgent()
    except Exception as e:
        logger.exception("Failed to initialize agents. Aborting.", exc_info=True)
        return

    # --- NEW: 2. Run Planner Agent ---
    logger.info("--- Invoking Planner Agent ---")
    plan_result = planner.run(initial_task)

    if not plan_result or not isinstance(plan_result.content, dict) or not plan_result.content.get('plan'):
        logger.error("--- Report Generation Failed: Planner Agent returned no valid plan data. ---")
        return

    logger.info(f"Plan generated: Language='{plan_result.content.get('language')}', Sections={len(plan_result.content.get('plan',[]))}")

    # --- NEW: 3. Instantiate and Run Orchestrator ---
    logger.info("--- Invoking Orchestrator ---")
    orchestrator = Orchestrator(
        research_agent=researcher,
        analysis_agent=analyzer,
        writer_agent=writer,
        chart_agent=chart_generator # Pass the chart agent instance
    )
    # --- MODIFIED: Use await ---
    sections_content = await orchestrator.run(initial_task=initial_task, planner_result=plan_result)

    if sections_content is None:
        logger.error("--- Report Generation Failed: Orchestrator did not complete successfully. ---")
        return

    logger.info("--- Orchestration Complete ---")


    # --- NEW: 4. Create Report Object ---
    logger.info("--- Assembling Final Report ---")
    try:
        final_report = Report(
            query=initial_task.query,
            plan=plan_result.content.get('plan', []),
            sections=sections_content if sections_content is not None else {},
        )
        logger.info("Report object created successfully.")
    except Exception as e:
        logger.exception("Failed to create Report object.", exc_info=True)
        return


    # --- MODIFIED: 5. Save Report as Markdown ---
    logger.info("--- Saving Report to Markdown File ---")
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        timestamp = final_report.generation_timestamp.strftime("%Y%m%d_%H%M%S")
        sanitized_query = "".join(c if c.isalnum() else "_" for c in final_report.query[:30]).strip('_')
        # Change extension to .md
        filename = f"report_{timestamp}_{sanitized_query}.md"
        filepath = os.path.join(REPORT_DIR, filename)

        # Get Markdown content from the report object
        markdown_content = final_report.to_markdown()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        logger.info(f"Report successfully saved to: {filepath}")
        print(f"\nReport saved to: {filepath}")

    except Exception as e:
        logger.error(f"Failed to save report to file: {e}", exc_info=True)
        print("\nError: Failed to save report to file.")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        # Using a query more suited for sections
        user_query = "Compare market share for phone brands in Q1 2024" # Query more likely to have data

    # --- MODIFIED: Use asyncio.run to call the async main function ---
    asyncio.run(run_report_generation(user_query))
