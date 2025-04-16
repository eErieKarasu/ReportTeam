from src.schemas import Task, Result
from src.core import generate_text
from src.actions.filter_extract_action import FilterAndExtractRelevantAction
from src.actions.synthesize_analyze_action import SynthesizeAnalyzeAndAssessAction
from typing import Optional, List, Dict, Any
import re
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

class AnalysisAgent:
    """
    使用 ReAct 模式分析研究结果，提取关键点，评估充分性，并尝试提取适合图表的结构化数据。
    """
    def __init__(self):
        # 初始化 Action 实例
        self.filter_action = FilterAndExtractRelevantAction()
        self.analyze_action = SynthesizeAnalyzeAndAssessAction()
    
    def _build_prompt(self, task: Task, research_result: Result, section_topic: str, language: str) -> str:
        """
        构建指导 ReAct 流程的提示
        
        Args:
            task: 原始任务
            research_result: 来自 ResearchAgent 的结果
            section_topic: 当前章节主题
            language: 目标语言
            
        Returns:
            ReAct 提示字符串
        """
        lang_name = {'zh': '中文', 'en': 'English'}.get(language, 'English')
        pages_count = len(research_result.content) if research_result and isinstance(research_result.content, list) else 0
        
        prompt = f"""你是一个分析专家，负责分析和综合研究结果，提取关键点、结构化数据，并评估信息充分性。

TASK: 针对用户查询 '{task.query}'，特别关注章节主题 '{section_topic}'，分析以下研究结果并生成一份综合报告。输出应为 {lang_name}。

BACKGROUND: 你收到了 {pages_count} 个网页的内容，这些是关于 '{task.query}' 的搜索结果，特别是关于 '{section_topic}' 的部分。

你可以使用以下 Actions:

1. **FilterAndExtractRelevant(research_results, section_topic)**
   - 描述: 对原始的研究结果列表进行初步筛选，提取与 '{section_topic}' 直接相关的内容片段或初步关键点。
   - 输入:
     - research_results: 网页内容列表
     - section_topic: 当前主题 '{section_topic}'
   - 输出: 提取出的初步关键点列表

2. **SynthesizeAnalyzeAndAssess(relevant_information, section_topic, original_query, language)**
   - 描述: 对筛选后的关键点进行综合分析，提炼最终关键观点，提取结构化数据，并评估信息充分性。
   - 输入:
     - relevant_information: 初步提取的关键点列表
     - section_topic: 当前主题 '{section_topic}'
     - original_query: 原始查询 '{task.query}'
     - language: 目标语言 '{language}'
   - 输出: 包含 key_points, structured_data, sufficiency, suggestion_for_further_research 的字典

3. **FinalAnswer(analysis_output)**
   - 描述: 返回最终的分析结果
   - 输入:
     - analysis_output: SynthesizeAnalyzeAndAssess 的结果
   - 输出: 最终分析结果

INSTRUCTIONS:
1. 你需要首先使用 FilterAndExtractRelevant 从原始研究结果中提取与 '{section_topic}' 相关的关键点。
2. 然后使用 SynthesizeAnalyzeAndAssess 进行深度分析和评估。
3. 最后使用 FinalAnswer 返回结果。

按照 Thought -> Action -> Observation 格式执行工作流程。每一步:
1. Thought: 思考当前情况和下一步行动。
2. Action: 选择并执行一个 Action，明确指定所有必要参数。
3. Observation: 记录 Action 的结果。

以下是一个示例:

研究结果: [包含3个页面的列表，省略具体内容]
章节主题: '人工智能应用'

Thought: 我需要从这些研究结果中提取与"人工智能应用"相关的关键点。我将使用 FilterAndExtractRelevant。

Action: FilterAndExtractRelevant(
  research_results=[网页内容列表],
  section_topic="人工智能应用"
)

Observation: [
  "人工智能在医疗领域的应用正在迅速增长，尤其是在诊断方面",
  "自动驾驶是人工智能的重要应用领域",
  "聊天机器人和虚拟助手利用自然语言处理技术"
]

Thought: 现在我有了初步的关键点，需要进行综合分析、提取结构化数据并评估信息充分性。

Action: SynthesizeAnalyzeAndAssess(
  relevant_information=[
    "人工智能在医疗领域的应用正在迅速增长，尤其是在诊断方面",
    "自动驾驶是人工智能的重要应用领域",
    "聊天机器人和虚拟助手利用自然语言处理技术"
  ],
  section_topic="人工智能应用",
  original_query="人工智能技术发展与应用",
  language="zh"
)

Observation: {{
  "sufficient": true,
  "key_points": [
    "人工智能在医疗诊断领域应用广泛，提高了诊断准确率",
    "自动驾驶技术是AI的重要应用方向，多家公司已实现L3级别自动驾驶",
    "智能助手和聊天机器人通过NLP技术提供个性化服务"
  ],
  "structured_data": [
    {{"应用领域": "医疗", "市场规模（十亿美元）": 45}},
    {{"应用领域": "自动驾驶", "市场规模（十亿美元）": 60}},
    {{"应用领域": "智能助手", "市场规模（十亿美元）": 25}}
  ],
  "suggestion_for_further_research": null
}}

Thought: 我已完成分析，获得了充分的信息和结构化数据。应返回最终结果。

Action: FinalAnswer(
  analysis_output={{
    "sufficient": true,
    "key_points": [
      "人工智能在医疗诊断领域应用广泛，提高了诊断准确率",
      "自动驾驶技术是AI的重要应用方向，多家公司已实现L3级别自动驾驶",
      "智能助手和聊天机器人通过NLP技术提供个性化服务"
    ],
    "structured_data": [
      {{"应用领域": "医疗", "市场规模（十亿美元）": 45}},
      {{"应用领域": "自动驾驶", "市场规模（十亿美元）": 60}},
      {{"应用领域": "智能助手", "市场规模（十亿美元）": 25}}
    ],
    "suggestion_for_further_research": null
  }}
)

现在，开始分析关于 '{section_topic}' 的研究结果:
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
        if action_name == "FilterAndExtractRelevant":
            research_results_match = re.search(r"research_results=(.+?)(?=,|\))", response, re.DOTALL)
            section_topic_match = re.search(r"section_topic=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)

            if research_results_match:
                action_params["research_results"] = "研究结果列表" # 实际实现时会传入真实数据
            if section_topic_match:
                action_params["section_topic"] = section_topic_match.group(1).strip()

        elif action_name == "SynthesizeAnalyzeAndAssess":
            relevant_info_match = re.search(r"relevant_information=(.+?)(?=,|\))", response, re.DOTALL)
            section_topic_match = re.search(r"section_topic=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            original_query_match = re.search(r"original_query=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            language_match = re.search(r"language=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)

            if relevant_info_match:
                action_params["relevant_information"] = "关键点列表" # 实际实现时会传入真实数据
            if section_topic_match:
                action_params["section_topic"] = section_topic_match.group(1).strip()
            if original_query_match:
                action_params["original_query"] = original_query_match.group(1).strip()
            if language_match:
                action_params["language"] = language_match.group(1).strip()

        elif action_name == "FinalAnswer":
            analysis_output_match = re.search(r"analysis_output=(.+?)(?=\))", response, re.DOTALL)
            if analysis_output_match:
                action_params["analysis_output"] = "分析结果" # 实际实现时会传入真实数据

        return {
            "thought": thought,
            "action": action_name,
            "action_input": action_params
        }

    async def run(self, original_task: Task, research_result: Result, section_topic: str, language: str) -> Optional[Result]:
        """
        执行分析任务：使用 ReAct 进行思考、行动和观察循环
        """
        logger.info(f"AnalysisAgent: 接收到关于任务 '{original_task.query}' 的研究结果 - 聚焦章节: '{section_topic}' (语言: {language})")
        
        if not research_result or not isinstance(research_result.content, list) or not research_result.content:
            logger.error("AnalysisAgent: 未收到有效的研究内容")
            return None
        
        # 构建初始 Prompt
        react_prompt = self._build_prompt(original_task, research_result, section_topic, language)
        
        # 保存中间结果
        intermediate_result = None
        
        # 执行 ReAct 循环
        max_iterations = 4  # 防止无限循环
        for iteration in range(max_iterations):
            logger.info(f"AnalysisAgent: 执行 ReAct 循环 (迭代 {iteration+1}/{max_iterations})")
            
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
            if action == "FilterAndExtractRelevant":
                observation = await self.filter_action.run(
                    research_results=research_result.content,
                    section_topic=section_topic
                )
                
            elif action == "SynthesizeAnalyzeAndAssess":
                if not intermediate_result:
                    logger.error("无法执行 SynthesizeAnalyzeAndAssess，因为缺少 FilterAndExtractRelevant 的结果")
                    # 尝试恢复
                    intermediate_result = await self.filter_action.run(
                        research_results=research_result.content,
                        section_topic=section_topic
                    )
                    
                observation = await self.analyze_action.run(
                    relevant_information=intermediate_result,
                    section_topic=section_topic,
                    original_query=original_task.query,
                    language=language
                )
                
            elif action == "FinalAnswer":
                # 检查是否已经有了最终分析结果
                if not observation and (not intermediate_result or isinstance(intermediate_result, list)):
                    logger.error("无法执行 FinalAnswer，因为缺少合适的分析结果")
                    return None
                    
                # 使用上一步的观察结果作为最终结果，确保是字典类型
                if isinstance(observation, dict):
                    final_result = observation
                elif isinstance(intermediate_result, dict):
                    final_result = intermediate_result
                else:
                    logger.error("FinalAnswer 需要字典类型的分析结果，但获得了列表类型")
                    return None
                    
                logger.info(f"AnalysisAgent: 任务完成，返回结果 (充分: {final_result.get('sufficient', False)})")
                return Result(content=final_result, source_agent="AnalysisAgent")
            
            else:
                logger.warning(f"未知的 Action: {action}")
                # 尝试恢复
                if not intermediate_result:
                    logger.info("尝试执行默认的 FilterAndExtractRelevant 操作")
                    intermediate_result = await self.filter_action.run(
                        research_results=research_result.content,
                        section_topic=section_topic
                    )
                    observation = intermediate_result
                    
            # 保存中间结果以备后用
            if action == "FilterAndExtractRelevant" and observation:
                intermediate_result = observation
            elif action == "SynthesizeAnalyzeAndAssess" and observation:
                intermediate_result = observation
            
            # 更新 Prompt 以包含最新的观察结果
            react_prompt += f"\n\nThought: {thought}\n\nAction: {action}({', '.join([f'{k}={v}' for k, v in action_input.items()])})\n\nObservation: {observation}\n\n"
        
        # 以下是循环外的代码
        logger.warning(f"AnalysisAgent: 达到最大迭代次数 ({max_iterations})，但未获得最终结果")
        
        # 如果达到最大迭代但有中间结果，尝试返回
        if isinstance(intermediate_result, dict) and "sufficient" in intermediate_result:
            logger.info("使用最后的分析结果作为最终结果")
            return Result(content=intermediate_result, source_agent="AnalysisAgent")
        
        return None


# 示例用法
if __name__ == '__main__':
    from src.utils import setup_logging
    setup_logging(log_level=logging.INFO)

    test_task = Task(query="比较2024年第一季度的手机品牌市场份额")
    # 模拟 ResearchAgent 输出格式的数据
    mock_research_data = Result(
        content=[
            {
                "url": "https://example.com/smartphone-market-q1",
                "content": """
                2024年第一季度智能手机市场表现强劲。苹果占据了全球市场份额的28%，
                较去年同期增长2个百分点。三星紧随其后，占据24%的市场份额，
                但较去年同期下降3个百分点。小米以15%的市场份额位居第三，
                OPPO和vivo分别占据10%和8%的市场份额。
                """
            },
            {
                "url": "https://example.com/tech-news",
                "content": """
                科技行业最新动态：高通发布了新一代骁龙处理器，将用于今年下半年发布的高端手机。
                特斯拉宣布新型电池技术突破，续航里程提升30%。
                """
            }
        ],
        source_agent="MockResearchAgent"
    )

    test_section = "手机市场份额分析"

    analyzer = AnalysisAgent()
    analysis_result = asyncio.run(analyzer.run(test_task, mock_research_data, section_topic=test_section, language='zh'))

    if analysis_result:
        print("\n--- AnalysisAgent 结果 ---")
        print(f"来源: {analysis_result.source_agent}")
        if isinstance(analysis_result.content, dict):
            content_dict = analysis_result.content
            print(f"信息充分: {content_dict.get('sufficient')}")
            print("关键点:")
            for point in content_dict.get('key_points', []):
                print(f"- {point}")

            structured_data_found = content_dict.get('structured_data')
            if structured_data_found:
                print("结构化数据:")
                print(json.dumps(structured_data_found, indent=2, ensure_ascii=False))
            else:
                print("结构化数据: 无")

            if not content_dict.get('sufficient'):
                print(f"建议: {content_dict.get('suggestion_for_further_research')}")
        else:
            print(f"意外的结果内容格式: {analysis_result.content}")
    else:
        print(f"\nAnalysisAgent 未能为章节 '{test_section}' 生成结果")
