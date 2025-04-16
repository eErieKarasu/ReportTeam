import logging
import asyncio
from typing import List, Dict, Optional, TYPE_CHECKING, Any
from src.schemas import Task, Result

# Use TYPE_CHECKING block for imports needed only for hints
if TYPE_CHECKING:
    from src.agents import ResearchAgent, AnalysisAgent, WriterAgent, ChartGenerationAgent

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Coordinates the workflow of agents, including targeted research based on
    AnalysisAgent's sufficiency assessment.
    """
    def __init__(self, research_agent: 'ResearchAgent', analysis_agent: 'AnalysisAgent', writer_agent: 'WriterAgent', chart_agent: 'ChartGenerationAgent'):
        """
        Initializes the Orchestrator with instances of the required agents.
        """
        self.research_agent = research_agent
        self.analysis_agent = analysis_agent
        self.writer_agent = writer_agent
        self.chart_agent = chart_agent
        logger.info("Orchestrator initialized with agents (including ChartGenerationAgent).")

    async def run(self, initial_task: 'Task', planner_result: Result) -> Optional[Dict[str, str]]:
        """
        Asynchronously executes the planned workflow: Global Research -> Loop(Analyze -> Targeted Research? -> Write).
        """
        if not planner_result or not isinstance(planner_result.content, dict):
             logger.error("Orchestrator received invalid planner result. Aborting.")
             return None
        planner_data: Dict[str, Any] = planner_result.content
        language = planner_data.get('language')
        plan = planner_data.get('plan')
        if not language or language not in ['en', 'zh'] or not plan or not isinstance(plan, list):
             logger.error("Orchestrator received invalid language or plan from planner. Aborting.")
             return None

        logger.info(f"Orchestrator starting run for task: '{initial_task.query}'")
        logger.info(f"Target Language: '{language}', Plan Sections: {len(plan)}")

        full_report_content: Dict[str, str] = {}

        logger.info("--- Running Global Research Phase (once) ---")
        global_research_result = await self.research_agent.run(initial_task)
        if not global_research_result:
            logger.error("Orchestrator: Global research failed. Cannot proceed. Aborting.")
            return None
        logger.info("--- Global Research Phase Complete ---")

        for i, section_topic in enumerate(plan):
            logger.info(f"--- Processing Section {i+1}/{len(plan)}: '{section_topic}' (Lang: {language}) ---")

            logger.debug(f"[{section_topic}] Invoking Analysis Agent (Initial Pass)...")
            initial_analysis_result = await self.analysis_agent.run(initial_task, global_research_result, section_topic, language=language)

            final_key_points_for_writer = None
            structured_data_for_charting = None
            chart_filepath = None
            section_failed = False

            if initial_analysis_result and isinstance(initial_analysis_result.content, dict):
                analysis_assessment = initial_analysis_result.content
                is_sufficient = analysis_assessment.get('sufficient', False) # Default to False if missing
                key_points = analysis_assessment.get('key_points', [])
                suggestion = analysis_assessment.get('suggestion_for_further_research')
                structured_data_for_charting = analysis_assessment.get('structured_data')

                logger.info(f"[{section_topic}] Initial Analysis Assessment: Sufficient={is_sufficient}, Points Found={len(key_points)}, StructuredData Found={'Yes' if structured_data_for_charting else 'No'}")

                if is_sufficient:
                    # Information is sufficient, use these key points
                    final_key_points_for_writer = key_points
                else:
                    # --- Information Insufficient: Trigger Targeted Research ---
                    logger.warning(f"[{section_topic}] Information deemed insufficient by AnalysisAgent.")
                    if suggestion:
                         logger.info(f"[{section_topic}] Suggestion for further research: {suggestion}")
                         # Construct targeted query (can be refined)
                         targeted_query_text = suggestion # Use suggestion directly
                         # Or combine: targeted_query_text = f"{initial_task.query} focusing on {section_topic}: {suggestion}"
                         targeted_task = Task(query=targeted_query_text)
                    else:
                         logger.warning(f"[{section_topic}] No specific suggestion provided, creating generic targeted query.")
                         # Fallback targeted query
                         targeted_task = Task(query=f"Detailed information about {section_topic} regarding {initial_task.query}")

                    logger.info(f"[{section_topic}] --- Running Targeted Research Phase ---")
                    logger.debug(f"[{section_topic}] Targeted Query: {targeted_task.query}")
                    targeted_research_result = await self.research_agent.run(targeted_task)

                    if not targeted_research_result:
                         logger.error(f"[{section_topic}] Targeted research failed. Proceeding with initial key points if any, but marking section potentially incomplete.")
                         # Use initial points, but the section might be bad.
                         final_key_points_for_writer = key_points # Use points found so far
                         if not final_key_points_for_writer:
                              logger.error(f"[{section_topic}] No initial key points found either. Cannot generate content.")
                              full_report_content[section_topic] = f"Error: Analysis failed for section '{section_topic}' (Initial and Targeted Research failed)."
                              section_failed = True
                         else:
                              logger.warning(f"[{section_topic}] Using potentially incomplete key points for writer after targeted research failure.")
                    else:
                         logger.info(f"[{section_topic}] --- Targeted Research Complete ---")
                         # --- Re-analyze using Targeted Research Results ---
                         # We need to decide if we pass *only* targeted results or combined results.
                         # Passing only targeted results might be cleaner for now.
                         logger.debug(f"[{section_topic}] Invoking Analysis Agent (Second Pass with Targeted Results)...")
                         # Note: We re-use the same analysis logic. It might report insufficient again,
                         # but we won't loop further in this implementation. We accept its output.
                         second_analysis_result = await self.analysis_agent.run(initial_task, targeted_research_result, section_topic, language=language)

                         if second_analysis_result and isinstance(second_analysis_result.content, dict):
                              second_assessment = second_analysis_result.content
                              logger.info(f"[{section_topic}] Second Analysis Assessment: Sufficient={second_assessment.get('sufficient')}, Points Found={len(second_assessment.get('key_points', []))}, StructuredData Found={'Yes' if second_assessment.get('structured_data') else 'No'}")
                              # Use the key points from the second analysis pass
                              final_key_points_for_writer = second_assessment.get('key_points', [])
                              # --- Update structured data from second pass ---
                              structured_data_for_charting = second_assessment.get('structured_data')
                              if not final_key_points_for_writer and not structured_data_for_charting: # Check if second pass yielded nothing useful
                                   logger.error(f"[{section_topic}] Second analysis pass yielded no key points or structured data.")
                                   # Fallback to initial points if available
                                   final_key_points_for_writer = key_points
                                   structured_data_for_charting = analysis_assessment.get('structured_data') # Use initial data too
                                   if not final_key_points_for_writer and not structured_data_for_charting:
                                        section_failed = True
                         else:
                              logger.error(f"[{section_topic}] Second analysis pass failed or returned invalid format. Using initial points if available.")
                              final_key_points_for_writer = key_points # Fallback to initial points
                              structured_data_for_charting = analysis_assessment.get('structured_data') # Use initial data
                              if not final_key_points_for_writer and not structured_data_for_charting: section_failed = True
                              else:
                                   logger.warning(f"[{section_topic}] Using potentially incomplete key points for writer after second analysis failure.")

            else: # Initial analysis failed or returned wrong format
                logger.error(f"[{section_topic}] Initial analysis failed or returned invalid format. Cannot generate content for this section.")
                full_report_content[section_topic] = f"Error: Analysis failed for section '{section_topic}' (Initial analysis)."
                section_failed = True

            # --- NEW: Attempt Chart Generation if data exists and section hasn't failed ---
            if not section_failed and structured_data_for_charting:
                logger.info(f"[{section_topic}] Attempting chart generation with extracted data.")
                # Create a mock Result object containing only the structured data for the chart agent
                chart_input_result = Result(content={'structured_data': structured_data_for_charting}, source_agent="AnalysisOutput")
                # ChartAgent.run is currently sync
                chart_result = self.chart_agent.run(analysis_result=chart_input_result, section_topic=section_topic)
                if chart_result and chart_result.content:
                     chart_filepath = chart_result.content # Store the path
                     logger.info(f"[{section_topic}] Chart generated successfully: {chart_filepath}")
                else:
                     logger.warning(f"[{section_topic}] Chart generation skipped or failed.")
                     chart_filepath = None # Ensure it's None if failed
            elif not section_failed:
                 logger.debug(f"[{section_topic}] No structured data found or suitable for charting.")

            # --- Proceed to Writing if we have key points ---
            if not section_failed and (final_key_points_for_writer is not None or chart_filepath is not None):
                 logger.debug(f"[{section_topic}] Invoking Writer Agent with {len(final_key_points_for_writer)} key points...")
                 # --- Create a mock Result object for the writer ---
                 # WriterAgent expects a Result object whose content is List[str]
                 writer_input_result = Result(content=final_key_points_for_writer if final_key_points_for_writer else [], source_agent="AnalysisAgentOutput")

                 section_content_result = self.writer_agent.run(
                     original_task=initial_task,
                     analysis_result=writer_input_result, # Pass the reconstructed Result
                     section_topic=section_topic,
                     language=language,
                     chart_filepath=chart_filepath # Pass the chart path
                 )

                 if section_content_result and section_content_result.content:
                     logger.info(f"Successfully generated content for section: '{section_topic}'")
                     full_report_content[section_topic] = section_content_result.content
                 else:
                     logger.error(f"[{section_topic}] Writing failed. Section content will be marked as failed.")
                     full_report_content[section_topic] = f"Error: Failed to generate content for section '{section_topic}'."
            elif not section_failed:
                 # This case happens if analysis (initial or second) succeeded but produced zero key points
                 logger.error(f"[{section_topic}] No key points or chart available for the writer. Cannot generate content.")
                 full_report_content[section_topic] = f"Error: No actionable content (points or chart) found for section '{section_topic}'."


        if not full_report_content:
             logger.warning("Orchestrator: No sections had content generated.")
        logger.info(f"Orchestrator finished processing all {len(plan)} sections.")
        return full_report_content
