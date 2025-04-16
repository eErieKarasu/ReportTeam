from src.schemas import Task, Result
from src.core import generate_text
from src.agents.planner_agent import PlannerAgent
from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.writer_agent import WriterAgent
from src.actions.call_planner_action import CallPlannerAgentAction
from src.actions.call_research_action import CallResearchAgentAction
from src.actions.call_analysis_action import CallAnalysisAgentAction
from src.actions.call_writer_action import CallWriterAgentAction
from src.actions.assemble_report_action import AssembleReportAction
from typing import Optional, List, Dict, Any
import re
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """
    使用 ReAct 模式协调各个 Agent 完成报告生成的主控 Agent。
    负责管理从任务规划到报告组装的整个工作流程。
    """
    def __init__(self):
        # 从core获取LLM客户端
        from src.core.llm_integration import get_llm_client
        from src.actions.create_plan_action import CreatePlanAction
        from src.actions.search_action import SearchAction
        from src.actions.browse_action import BrowseAction
        from src.actions.synthesize_analyze_action import SynthesizeAnalyzeAndAssessAction
        from src.actions.write_section_action import WriteSectionAction
        
        # 获取LLM客户端
        self.llm_client = get_llm_client()
        logger.info(f"OrchestratorAgent: LLM client initialized: {type(self.llm_client).__name__}")
        
        # 初始化各个Action
        # PlannerAgent相关
        self.create_plan_action = CreatePlanAction()
        
        # ResearchAgent相关
        self.search_action = SearchAction()
        self.browse_action = BrowseAction()
        
        # 初始化各个子Agent
        self.planner_agent = PlannerAgent(
            llm_client=self.llm_client,
            actions=[self.create_plan_action],
        )
        
        self.research_agent = ResearchAgent(
            llm_client=self.llm_client,
            actions=[self.search_action, self.browse_action],
        )
        
        # AnalysisAgent和WriterAgent不需要传递llm_client和actions
        self.analysis_agent = AnalysisAgent()
        self.writer_agent = WriterAgent()
        
        # 初始化Action实例
        self.call_planner_action = CallPlannerAgentAction(self.planner_agent)
        self.call_research_action = CallResearchAgentAction(self.research_agent)
        self.call_analysis_action = CallAnalysisAgentAction(self.analysis_agent)
        self.call_writer_action = CallWriterAgentAction(self.writer_agent)
        self.assemble_report_action = AssembleReportAction()
    
    def _build_prompt(self, task: Task, current_state: Dict[str, Any], language: str) -> str:
        """
        构建指导 ReAct 流程的提示
        
        Args:
            task: 用户任务
            current_state: 当前状态字典，包括以下可能的键:
                - plan: 报告计划（章节列表等）
                - processed_sections: 已处理的章节及其状态
                - current_section: 当前正在处理的章节
                - in_progress: 是否有正在进行的工作流程
            language: 目标语言
            
        Returns:
            ReAct 提示字符串
        """
        lang_name = {'zh': '中文', 'en': 'English'}.get(language, 'English')
        
        # 提取当前状态信息
        plan = current_state.get('plan', {})
        
        # 确保plan是字典，如果是None则初始化为空字典
        if plan is None:
            plan = {}
            
        sections = plan.get('sections', [])
        processed_sections = current_state.get('processed_sections', {})
        current_section = current_state.get('current_section', None)
        
        # 计算进度信息
        total_sections = len(sections)
        completed_sections = sum(1 for s in processed_sections.values() if s.get('status') == 'completed')
        completion_percent = int(completed_sections/total_sections*100) if total_sections else 0
        
        prompt = f"""你是一个专业的报告项目经理，负责协调各个专家团队完成一份高质量报告。

TASK: 根据用户请求 '{task.query}' 生成一份完整的 {lang_name} 报告。

PROGRESS SUMMARY:
- {'已有报告计划' if plan else '尚未获取报告计划'}
- {'已有章节列表: ' + ', '.join(sections) if sections else '尚未确定章节列表'}
- 总进度: {completed_sections}/{total_sections} 章节完成 ({completion_percent}%)
- 当前处理: {current_section if current_section else '无'}

你可以使用以下 Actions:

1. **CallPlannerAgent(task)**
   - 描述: 调用规划专家将用户任务分解为报告章节大纲
   - 输入:
     - task: 用户任务
   - 输出: 包含章节列表的计划

2. **CallResearchAgent(section_topic, original_query, language)**
   - 描述: 调用研究专家为指定章节搜集信息
   - 输入:
     - section_topic: 章节主题，例如 '{sections[0] if sections else "章节主题"}'
     - original_query: 原始查询 '{task.query}'
     - language: 目标语言 '{language}'
   - 输出: 包含研究结果的字典

3. **CallAnalysisAgent(research_result, section_topic, original_query, language)**
   - 描述: 调用分析专家分析指定章节的研究结果
   - 输入:
     - research_result: ResearchAgent 的结果
     - section_topic: 章节主题
     - original_query: 原始查询 '{task.query}'
     - language: 目标语言 '{language}'
   - 输出: 包含分析结果的字典

4. **CallWriterAgent(analysis_result, section_topic, original_query, language)**
   - 描述: 调用写作专家根据分析结果撰写指定章节的文本
   - 输入:
     - analysis_result: AnalysisAgent 的结果
     - section_topic: 章节主题
     - original_query: 原始查询 '{task.query}'
     - language: 目标语言 '{language}'
   - 输出: 包含章节文本的字典

5. **AssembleReport(written_sections, section_order, report_title, language)**
   - 描述: 将所有已撰写好的章节文本按计划顺序组装成最终报告
   - 输入:
     - written_sections: 章节标题到章节内容的映射
     - section_order: 章节顺序列表
     - report_title: 报告标题
     - language: 报告语言
   - 输出: 最终组装好的报告文本

6. **FinalAnswer(final_report_text)**
   - 描述: 返回最终的报告文本
   - 输入:
     - final_report_text: 完整的报告文本
   - 输出: 最终报告

INSTRUCTIONS:
1. 始终按照以下工作流程: 规划 -> 各章节(研究->分析->撰写) -> 组装
2. 先调用 PlannerAgent 获取报告计划
3. 然后按顺序处理每个章节，每个章节依次调用 ResearchAgent -> AnalysisAgent -> WriterAgent
4. 所有章节处理完成后，调用 AssembleReport 组装报告
5. 最后使用 FinalAnswer 返回结果

按照 Thought -> Action -> Observation 格式执行工作流程。每一步:
1. Thought: 思考当前情况和下一步行动
2. Action: 选择并执行一个 Action，明确指定所有必要参数
3. Observation: 记录 Action 的结果

以下是一个示例工作流:

用户任务: "人工智能在各行业的应用"

Thought: 首先需要规划整个报告的结构，确定各个章节。我将调用 PlannerAgent。

Action: CallPlannerAgent(
  task={{query: "人工智能在各行业的应用"}}
)

Observation: {{
  "success": true,
  "sections": ["人工智能简介", "医疗行业的AI应用", "金融行业的AI应用", "教育领域的AI应用", "总结与展望"],
  "full_plan": {{...}}
}}

Thought: 已获得报告计划，包含5个章节。现在开始处理第一个章节"人工智能简介"。首先需要为该章节搜集信息。

Action: CallResearchAgent(
  section_topic="人工智能简介",
  original_query="人工智能在各行业的应用",
  language="zh"
)

Observation: {{
  "success": true,
  "section_topic": "人工智能简介",
  "result": {{...}} // ResearchAgent 的结果
}}

Thought: 已获得"人工智能简介"章节的研究结果。现在需要分析这些结果。

Action: CallAnalysisAgent(
  research_result=上一步的 result,
  section_topic="人工智能简介",
  original_query="人工智能在各行业的应用",
  language="zh"
)

Observation: {{
  "success": true,
  "section_topic": "人工智能简介",
  "result": {{...}}, // AnalysisAgent 的结果
  "sufficient": true
}}

Thought: 已获得"人工智能简介"章节的分析结果，信息充分。现在可以撰写该章节的内容。

Action: CallWriterAgent(
  analysis_result=上一步的 result,
  section_topic="人工智能简介",
  original_query="人工智能在各行业的应用",
  language="zh"
)

Observation: {{
  "success": true,
  "section_topic": "人工智能简介",
  "result": {{"content": "人工智能(AI)是计算机科学的一个分支，致力于开发能够执行通常需要人类智能的任务的系统..."}} // WriterAgent 的结果
}}

// ... (对剩余章节重复相同过程) ...

Thought: 所有章节都已完成撰写。现在需要将它们组装成最终报告。

Action: AssembleReport(
  written_sections={{"人工智能简介": "...", "医疗行业的AI应用": "...", ...}},
  section_order=["人工智能简介", "医疗行业的AI应用", "金融行业的AI应用", "教育领域的AI应用", "总结与展望"],
  report_title="人工智能在各行业的应用报告",
  language="zh"
)

Observation: "# 人工智能在各行业的应用报告\n\n## 人工智能简介\n人工智能(AI)是计算机科学的一个分支，致力于开发能够执行通常需要人类智能的任务的系统...\n\n## 医疗行业的AI应用\n..."

Thought: 报告已成功组装。现在我可以返回最终结果。

Action: FinalAnswer(
  final_report_text="# 人工智能在各行业的应用报告\n\n## 人工智能简介\n人工智能(AI)是计算机科学的一个分支，致力于开发能够执行通常需要人类智能的任务的系统...\n\n## 医疗行业的AI应用\n..."
)

现在，请根据当前状态，为任务 '{task.query}' 确定下一步行动:
"""
        return prompt
    
    def _parse_llm_react_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 的 ReAct 响应
        
        Args:
            response: LLM 的原始响应文本
            
        Returns:
            包含 thought, action, action_input 的字典
        """
        # 提取 Thought
        thought_match = re.search(r"Thought:(.+?)(?=Action:|$)", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # 提取 Action 和 action_input
        action_match = re.search(r"Action:(.+?)(?=\(|\n|$)", response, re.DOTALL)
        action_name = action_match.group(1).strip() if action_match else ""
        
        # 提取 Action 的参数
        action_params = {}
        
        if action_name == "CallPlannerAgent":
            # CallPlannerAgent 的参数相对简单，主要是 task
            action_params["task"] = "USER_TASK" # 实际执行时会传入真实任务
            
        elif action_name == "CallResearchAgent":
            section_topic_match = re.search(r"section_topic=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            original_query_match = re.search(r"original_query=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            language_match = re.search(r"language=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            
            if section_topic_match:
                action_params["section_topic"] = section_topic_match.group(1).strip()
            if original_query_match:
                action_params["original_query"] = original_query_match.group(1).strip()
            if language_match:
                action_params["language"] = language_match.group(1).strip()
                
        elif action_name == "CallAnalysisAgent":
            research_result_match = re.search(r"research_result=(.+?)(?=,|\))", response, re.DOTALL)
            section_topic_match = re.search(r"section_topic=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            original_query_match = re.search(r"original_query=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            language_match = re.search(r"language=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            
            if research_result_match:
                action_params["research_result"] = "PREVIOUS_RESULT" # 实际执行时会传入真实结果
            if section_topic_match:
                action_params["section_topic"] = section_topic_match.group(1).strip()
            if original_query_match:
                action_params["original_query"] = original_query_match.group(1).strip()
            if language_match:
                action_params["language"] = language_match.group(1).strip()
                
        elif action_name == "CallWriterAgent":
            analysis_result_match = re.search(r"analysis_result=(.+?)(?=,|\))", response, re.DOTALL)
            section_topic_match = re.search(r"section_topic=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            original_query_match = re.search(r"original_query=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            language_match = re.search(r"language=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            
            if analysis_result_match:
                action_params["analysis_result"] = "PREVIOUS_RESULT" # 实际执行时会传入真实结果
            if section_topic_match:
                action_params["section_topic"] = section_topic_match.group(1).strip()
            if original_query_match:
                action_params["original_query"] = original_query_match.group(1).strip()
            if language_match:
                action_params["language"] = language_match.group(1).strip()
                
        elif action_name == "AssembleReport":
            written_sections_match = re.search(r"written_sections=(.+?)(?=,|\))", response, re.DOTALL)
            section_order_match = re.search(r"section_order=(.+?)(?=,|\))", response, re.DOTALL)
            report_title_match = re.search(r"report_title=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            language_match = re.search(r"language=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            
            if written_sections_match:
                action_params["written_sections"] = "WRITTEN_SECTIONS" # 实际执行时会传入真实章节
            if section_order_match:
                action_params["section_order"] = "SECTION_ORDER" # 实际执行时会传入真实顺序
            if report_title_match:
                action_params["report_title"] = report_title_match.group(1).strip()
            if language_match:
                action_params["language"] = language_match.group(1).strip()
                
        elif action_name == "FinalAnswer":
            final_report_match = re.search(r"final_report_text=(.+?)(?=\))", response, re.DOTALL)
            if final_report_match:
                action_params["final_report_text"] = "FINAL_REPORT" # 实际执行时会传入真实报告
        
        return {
            "thought": thought,
            "action": action_name,
            "action_input": action_params
        }
    
    async def run(self, task: Task, language: str = "zh") -> Optional[Result]:
        """
        执行报告生成任务：使用 ReAct 进行思考、行动和观察循环
        
        Args:
            task: 用户任务，包含查询
            language: 目标语言，默认为中文
            
        Returns:
            Result 对象，其中 content 是最终的报告文本，或 None（如果生成失败）
        """
        logger.info(f"OrchestratorAgent: 开始处理任务 '{task.query}' (语言: {language})")
        
        # 初始化状态
        state = {
            'plan': None,  # 计划 (包含章节列表)
            'processed_sections': {},  # 已处理章节的状态
            'current_section': None,  # 当前处理的章节
            'section_contents': {},  # 章节内容 {章节名: 内容}
            'temp_results': {}  # 临时存储 Agent 返回结果 {章节名_阶段: 结果}
        }
        
        # 构建初始 Prompt
        react_prompt = self._build_prompt(task, state, language)
        
        # 执行 ReAct 循环
        max_iterations = 30  # 防止无限循环
        for iteration in range(max_iterations):
            logger.info(f"OrchestratorAgent: 执行 ReAct 循环 (迭代 {iteration+1}/{max_iterations})")
            
            # 调用 LLM 获取下一步行动
            llm_response = generate_text(react_prompt)
            logger.debug(f"LLM ReAct 响应:\n{llm_response}")
            
            # 解析 LLM 响应
            parsed_response = self._parse_llm_react_response(llm_response)
            thought = parsed_response["thought"]
            action = parsed_response["action"]
            action_input = parsed_response["action_input"]
            
            logger.info(f"Thought: {thought[:100]}...")
            logger.info(f"选择的 Action: {action}")
            
            # 执行 Action
            observation = None
            
            if action == "CallPlannerAgent":
                # 调用 PlannerAgent
                planner_result = await self.call_planner_action.run(task=task)
                observation = planner_result
                
                # 更新状态
                if planner_result.get('success', False):
                    state['plan'] = planner_result
                    # 初始化所有章节的状态
                    for section in planner_result.get('sections', []):
                        state['processed_sections'][section] = {'status': 'pending'}
                
            elif action == "CallResearchAgent":
                # 提取参数
                section_topic = action_input.get('section_topic')
                original_query = action_input.get('original_query', task.query)
                language_param = action_input.get('language', language)
                
                if not section_topic:
                    logger.error("CallResearchAgent 缺少必要参数: section_topic")
                    observation = {"success": False, "error": "缺少章节主题"}
                else:
                    # 更新当前处理的章节
                    state['current_section'] = section_topic
                    
                    # 调用 ResearchAgent
                    research_result = await self.call_research_action.run(
                        section_topic=section_topic,
                        original_query=original_query,
                        language=language_param
                    )
                    observation = research_result
                    
                    # 更新状态
                    if research_result.get('success', False):
                        # 保存研究结果
                        state['temp_results'][f"{section_topic}_research"] = research_result.get('result')
                        # 更新章节状态
                        if section_topic in state['processed_sections']:
                            state['processed_sections'][section_topic]['research_done'] = True
                
            elif action == "CallAnalysisAgent":
                # 提取参数
                section_topic = action_input.get('section_topic')
                original_query = action_input.get('original_query', task.query)
                language_param = action_input.get('language', language)
                
                if not section_topic:
                    logger.error("CallAnalysisAgent 缺少必要参数: section_topic")
                    observation = {"success": False, "error": "缺少章节主题"}
                else:
                    # 获取该章节的研究结果
                    research_result = state['temp_results'].get(f"{section_topic}_research")
                    
                    if not research_result:
                        logger.error(f"无法找到章节 '{section_topic}' 的研究结果")
                        observation = {"success": False, "error": "缺少研究结果"}
                    else:
                        # 调用 AnalysisAgent
                        analysis_result = await self.call_analysis_action.run(
                            research_result=research_result,
                            section_topic=section_topic,
                            original_query=original_query,
                            language=language_param
                        )
                        observation = analysis_result
                        
                        # 更新状态
                        if analysis_result.get('success', False):
                            # 保存分析结果
                            state['temp_results'][f"{section_topic}_analysis"] = analysis_result.get('result')
                            # 更新章节状态
                            if section_topic in state['processed_sections']:
                                state['processed_sections'][section_topic]['analysis_done'] = True
                                state['processed_sections'][section_topic]['analysis_sufficient'] = analysis_result.get('sufficient', False)
                
            elif action == "CallWriterAgent":
                # 提取参数
                section_topic = action_input.get('section_topic')
                original_query = action_input.get('original_query', task.query)
                language_param = action_input.get('language', language)
                
                if not section_topic:
                    logger.error("CallWriterAgent 缺少必要参数: section_topic")
                    observation = {"success": False, "error": "缺少章节主题"}
                else:
                    # 获取该章节的分析结果
                    analysis_result = state['temp_results'].get(f"{section_topic}_analysis")
                    
                    if not analysis_result:
                        logger.error(f"无法找到章节 '{section_topic}' 的分析结果")
                        observation = {"success": False, "error": "缺少分析结果"}
                    else:
                        # 调用 WriterAgent
                        writer_result = await self.call_writer_action.run(
                            analysis_result=analysis_result,
                            section_topic=section_topic,
                            original_query=original_query,
                            language=language_param
                        )
                        observation = writer_result
                        
                        # 更新状态
                        if writer_result.get('success', False):
                            # 保存写作结果
                            written_content = writer_result.get('result').content if writer_result.get('result') else ""
                            state['section_contents'][section_topic] = written_content
                            # 更新章节状态
                            if section_topic in state['processed_sections']:
                                state['processed_sections'][section_topic]['status'] = 'completed'
                                state['processed_sections'][section_topic]['writing_done'] = True
                
            elif action == "AssembleReport":
                # 提取参数
                report_title = action_input.get('report_title', f"{task.query} 报告")
                language_param = action_input.get('language', language)
                
                # 从状态中获取已写好的章节和章节顺序
                written_sections = state.get('section_contents', {})
                section_order = state.get('plan', {}).get('sections', [])
                
                if not section_order:
                    logger.error("AssembleReport 找不到章节顺序")
                    observation = {"success": False, "error": "缺少章节顺序"}
                elif not written_sections:
                    logger.error("AssembleReport 找不到章节内容")
                    observation = {"success": False, "error": "缺少章节内容"}
                else:
                    # 调用 AssembleReport
                    assembled_report = await self.assemble_report_action.run(
                        written_sections=written_sections,
                        section_order=section_order,
                        report_title=report_title,
                        language=language_param
                    )
                    
                    # 保存最终报告
                    state['final_report'] = assembled_report
                    observation = assembled_report[:500] + "..." if len(assembled_report) > 500 else assembled_report
                
            elif action == "FinalAnswer":
                # 获取最终报告
                final_report = state.get('final_report')
                
                if not final_report:
                    logger.error("FinalAnswer 找不到最终报告")
                    observation = {"success": False, "error": "缺少最终报告"}
                else:
                    logger.info(f"OrchestratorAgent: 任务完成，返回最终报告 ({len(final_report)} 字符)")
                    return Result(content=final_report, source_agent="OrchestratorAgent")
            
            else:
                logger.warning(f"未知的 Action: {action}")
                observation = {"success": False, "error": f"未知的动作: {action}"}
            
            # 更新 Prompt 以包含最新的观察结果和状态
            react_prompt = self._build_prompt(task, state, language)
            react_prompt += f"\n\nThought: {thought}\n\nAction: {action}({', '.join([f'{k}={v}' for k, v in action_input.items()])})\n\nObservation: {observation}\n\n"
        
        # 如果达到最大迭代次数但有最终报告，返回它
        final_report = state.get('final_report')
        if final_report:
            logger.info(f"OrchestratorAgent: 达到最大迭代次数但有最终报告，返回结果 ({len(final_report)} 字符)")
            return Result(content=final_report, source_agent="OrchestratorAgent")
        
        # 如果所有章节都已完成，尝试组装一个简单的报告
        if state['plan'] and all(s.get('status') == 'completed' for s in state['processed_sections'].values()):
            logger.info("OrchestratorAgent: 所有章节都已完成，尝试组装简单报告")
            try:
                # 从状态中获取已写好的章节和章节顺序
                written_sections = state.get('section_contents', {})
                section_order = state.get('plan', {}).get('sections', [])
                report_title = f"{task.query} 报告"
                
                # 组装简单报告
                simple_report = await self.assemble_report_action.run(
                    written_sections=written_sections,
                    section_order=section_order,
                    report_title=report_title,
                    language=language
                )
                
                return Result(content=simple_report, source_agent="OrchestratorAgent")
            except Exception as e:
                logger.error(f"组装简单报告时出错: {e}")
        
        logger.error(f"OrchestratorAgent: 达到最大迭代次数 ({max_iterations})，但未生成最终报告")
        return None


# 示例用法
if __name__ == '__main__':
    from src.utils import setup_logging
    setup_logging(log_level=logging.INFO)

    test_task = Task(query="生成一份关于大语言模型发展的报告，包含三个章节")
    test_language = "zh"
    
    orchestrator = OrchestratorAgent()
    report_result = asyncio.run(orchestrator.run(test_task, test_language))

    if report_result:
        print("\n--- 生成的报告 ---")
        print(f"来源: {report_result.source_agent}")
        print(f"内容 ({len(report_result.content)} 字符):")
        print(report_result.content[:500] + "..." if len(report_result.content) > 500 else report_result.content)
        
        # 保存报告到文件
        import os
        os.makedirs("output/reports", exist_ok=True)
        report_filename = f"output/reports/{test_task.query.replace(' ', '_')}.md"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_result.content)
        print(f"\n报告已保存到: {report_filename}")
    else:
        print("\nOrchestratorAgent 未能生成报告")