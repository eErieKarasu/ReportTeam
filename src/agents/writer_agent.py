from src.schemas import Task, Result
from src.core import generate_text
from src.actions.write_section_action import WriteSectionAction
from typing import Optional, List, Dict, Any
import re
import logging
import asyncio

logger = logging.getLogger(__name__)

class WriterAgent:
    """
    使用 ReAct 模式将分析结果转化为连贯文本的 Agent。
    """
    def __init__(self):
        # 初始化 Action 实例
        self.write_section_action = WriteSectionAction()
    
    def _build_prompt(self, task: Task, analysis_result: Result, section_topic: str, language: str) -> str:
        """
        构建指导 ReAct 流程的提示
        
        Args:
            task: 原始任务
            analysis_result: 分析结果
            section_topic: 当前章节主题
            language: 目标语言
            
        Returns:
            ReAct 提示字符串
        """
        lang_name = {'zh': '中文', 'en': 'English'}.get(language, 'English')
        
        # 提取分析结果中的关键信息（假设 analysis_result.content 是字典）
        content_dict = analysis_result.content if isinstance(analysis_result.content, dict) else {}
        key_points = content_dict.get('key_points', [])
        structured_data = content_dict.get('structured_data', [])
        is_sufficient = content_dict.get('sufficient', False)
        suggestion = content_dict.get('suggestion_for_further_research', "")
        
        key_points_count = len(key_points) if isinstance(key_points, list) else 0
        has_structured_data = structured_data is not None and len(structured_data) > 0 if isinstance(structured_data, list) else False
        
        prompt = f"""你是一名专业内容撰写者，负责将分析结果转化为高质量的报告章节。

TASK: 针对用户查询 '{task.query}'，为章节 '{section_topic}' 撰写流畅、专业的文本内容。输出应为 {lang_name}。

BACKGROUND: 你收到了分析结果，包含 {key_points_count} 个关键点{"和结构化数据" if has_structured_data else ""}。信息充分性: {"充分" if is_sufficient else "不充分"}。

你可以使用以下 Actions:

1. **WriteSection(key_points, structured_data, section_topic, original_query, language)**
   - 描述: 根据提供的关键点和结构化数据，撰写指定章节的流畅文本
   - 输入:
     - key_points: 关键点列表 {key_points if key_points_count <= 3 else str(key_points[:2]) + "...等"}
     - structured_data: 结构化数据 {structured_data if has_structured_data and len(structured_data) <= 2 else "..."}
     - section_topic: 当前章节 '{section_topic}'
     - original_query: 原始查询 '{task.query}'
     - language: 目标语言 '{language}'
   - 输出: 生成的章节文本

2. **FinalAnswer(final_text)**
   - 描述: 返回最终的章节文本
   - 输入:
     - final_text: WriteSection 生成的文本
   - 输出: 最终章节文本

INSTRUCTIONS:
1. 你需要使用 WriteSection 根据分析结果撰写章节内容
2. 然后使用 FinalAnswer 返回结果

按照 Thought -> Action -> Observation 格式执行工作流程。每一步:
1. Thought: 思考当前情况和下一步行动
2. Action: 选择并执行一个 Action，明确指定所有必要参数
3. Observation: 记录 Action 的结果

以下是一个示例:

任务: 为"人工智能在医疗领域的应用"章节撰写内容

Thought: 我需要根据提供的关键点和结构化数据，为"人工智能在医疗领域的应用"章节撰写流畅的文本。我将使用 WriteSection。

Action: WriteSection(
  key_points=[
    "人工智能在医疗诊断领域应用广泛，提高了诊断准确率",
    "机器学习算法可以分析医学影像，帮助识别癌症等疾病",
    "AI辅助系统减少了医生的工作负担，提高了医疗效率"
  ],
  structured_data=[
    {{"应用类型": "诊断辅助", "采用率": "76%"}},
    {{"应用类型": "影像分析", "采用率": "82%"}}
  ],
  section_topic="人工智能在医疗领域的应用",
  original_query="人工智能技术在各行业的应用",
  language="zh"
)

Observation: "人工智能技术正在彻底改变医疗领域的诊断和治疗方式。在诊断方面，AI系统已经展现出超越传统方法的准确率，尤其是在复杂病例的早期识别上。医学影像分析是AI最成功的应用领域之一，机器学习算法能够迅速分析CT、MRI等影像资料，精确识别癌症等疾病的早期迹象，为医生提供可靠的诊断参考。

此外，AI辅助系统显著减轻了医生的工作负担，提高了整体医疗效率。通过自动化处理常规任务和初步筛查，医生可以将更多精力集中在复杂病例和患者护理上。目前，诊断辅助系统在医疗机构的采用率已达76%，而影像分析技术的采用率更是高达82%，显示出医疗行业对AI技术的广泛接受。"

Thought: 我已经成功为"人工智能在医疗领域的应用"章节生成了流畅、专业的文本，涵盖了所有关键点和结构化数据。现在应该返回这个结果。

Action: FinalAnswer(
  final_text="人工智能技术正在彻底改变医疗领域的诊断和治疗方式。在诊断方面，AI系统已经展现出超越传统方法的准确率，尤其是在复杂病例的早期识别上。医学影像分析是AI最成功的应用领域之一，机器学习算法能够迅速分析CT、MRI等影像资料，精确识别癌症等疾病的早期迹象，为医生提供可靠的诊断参考。

此外，AI辅助系统显著减轻了医生的工作负担，提高了整体医疗效率。通过自动化处理常规任务和初步筛查，医生可以将更多精力集中在复杂病例和患者护理上。目前，诊断辅助系统在医疗机构的采用率已达76%，而影像分析技术的采用率更是高达82%，显示出医疗行业对AI技术的广泛接受。"
)

现在，开始为章节 '{section_topic}' 撰写内容:
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
        if action_name == "WriteSection":
            key_points_match = re.search(r"key_points=(.+?)(?=,|\))", response, re.DOTALL)
            structured_data_match = re.search(r"structured_data=(.+?)(?=,|\))", response, re.DOTALL)
            section_topic_match = re.search(r"section_topic=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            original_query_match = re.search(r"original_query=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            language_match = re.search(r"language=\"?([^\"]+?)\"?(?=\)|,)", response, re.DOTALL)
            
            if key_points_match:
                action_params["key_points"] = "关键点列表" # 实际执行时会传入真实数据
            if structured_data_match:
                action_params["structured_data"] = "结构化数据" # 实际执行时会传入真实数据
            if section_topic_match:
                action_params["section_topic"] = section_topic_match.group(1).strip()
            if original_query_match:
                action_params["original_query"] = original_query_match.group(1).strip()
            if language_match:
                action_params["language"] = language_match.group(1).strip()
                
        elif action_name == "FinalAnswer":
            final_text_match = re.search(r"final_text=(.+?)(?=\))", response, re.DOTALL)
            if final_text_match:
                action_params["final_text"] = "最终文本" # 实际执行时会传入真实数据
        
        return {
            "thought": thought,
            "action": action_name,
            "action_input": action_params
        }
    
    async def run(self, 
            original_task: Task,
            analysis_result: Result,
            section_topic: str,
            language: str,
            chart_filepath: Optional[str] = None) -> Optional[Result]:
        """
        执行写作任务：使用 ReAct 进行思考、行动和观察循环

        Args:
            original_task: 原始任务
            analysis_result: 来自 AnalysisAgent 的结果，content 是一个包含 key_points 等的字典
            section_topic: 当前章节主题
            language: 目标语言
            chart_filepath: 图表文件路径（可选）

        Returns:
            Result 对象，其中 content 是生成的章节文本，或 None（如果生成失败）
        """
        logger.info(f"WriterAgent: 接收到关于任务 '{original_task.query}' 的分析结果 - 章节: '{section_topic}' (语言: {language})")

        # 检查输入
        if not analysis_result or not isinstance(analysis_result.content, dict) or not analysis_result.content.get('key_points'):
            logger.error("WriterAgent: 未收到有效的分析内容")
            return None

        # 提取分析结果中的信息
        content_dict = analysis_result.content
        key_points = content_dict.get('key_points', [])
        structured_data = content_dict.get('structured_data')
        
        logger.info(f"WriterAgent: 收到 {len(key_points)} 个关键点和 {'有' if structured_data else '无'} 结构化数据")
        
        # 构建初始 Prompt
        react_prompt = self._build_prompt(original_task, analysis_result, section_topic, language)
        
        # 保存生成的文本
        generated_text = None
        
        # 执行 ReAct 循环
        max_iterations = 3  # 防止无限循环
        for iteration in range(max_iterations):
            logger.info(f"WriterAgent: 执行 ReAct 循环 (迭代 {iteration+1}/{max_iterations})")
            
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
            if action == "WriteSection":
                observation = await self.write_section_action.run(
                    key_points=key_points,
                    structured_data=structured_data,
                    section_topic=section_topic,
                    original_query=original_task.query,
                    language=language,
                    chart_filepath=chart_filepath
                )
                # 保存生成的文本
                generated_text = observation
                
            elif action == "FinalAnswer":
                if not generated_text:
                    logger.error("无法执行 FinalAnswer，因为缺少生成的文本")
                    return None
                
                logger.info(f"WriterAgent: 任务完成，返回生成的文本 ({len(generated_text)} 字符)")
                return Result(content=generated_text, source_agent="WriterAgent")
            
            else:
                logger.warning(f"未知的 Action: {action}")
                # 如果 LLM 响应不包含有效的 Action，尝试直接生成文本
                if not generated_text:
                    logger.info("尝试直接生成文本")
                    generated_text = await self.write_section_action.run(
                        key_points=key_points,
                        structured_data=structured_data,
                        section_topic=section_topic,
                        original_query=original_task.query,
                        language=language,
                        chart_filepath=chart_filepath
                    )
                    observation = generated_text
            
            # 更新 Prompt 以包含最新的观察结果
            react_prompt += f"\n\nThought: {thought}\n\nAction: {action}({', '.join([f'{k}={v}' for k, v in action_input.items()])})\n\nObservation: {observation}\n\n"
        
        # 如果达到最大迭代但有生成的文本，返回它
        if generated_text:
            logger.info(f"WriterAgent: 达到最大迭代次数但有生成文本，返回结果 ({len(generated_text)} 字符)")
            return Result(content=generated_text, source_agent="WriterAgent")
        
        logger.error(f"WriterAgent: 达到最大迭代次数 ({max_iterations})，但未生成文本")

        return None


# 示例用法
if __name__ == '__main__':
    from src.utils import setup_logging
    from src.schemas import Task, Result
    setup_logging(log_level=logging.INFO)

    test_task = Task(query="比较2024年第一季度的手机品牌市场份额")
    # 模拟 AnalysisAgent 的 ReAct 模式输出
    mock_analysis_data = Result(
        content={
            "sufficient": True,
            "key_points": [
                "2024年第一季度智能手机市场表现强劲",
                "苹果以28%的市场份额位居全球第一，较去年同期增长2个百分点",
                "三星以24%的市场份额排名第二，但较去年同期下降3个百分点",
                "小米以15%的市场份额排名第三",
                "OPPO和vivo分别占据10%和8%的市场份额"
            ],
            "structured_data": [
                {"品牌": "苹果", "市场份额": 28},
                {"品牌": "三星", "市场份额": 24},
                {"品牌": "小米", "市场份额": 15},
                {"品牌": "OPPO", "市场份额": 10},
                {"品牌": "vivo", "市场份额": 8}
            ],
            "suggestion_for_further_research": None
        },
        source_agent="AnalysisAgent"
    )

    test_section = "手机市场份额分析"

    writer = WriterAgent()
    writer_result = asyncio.run(writer.run(test_task, mock_analysis_data, section_topic=test_section, language='zh'))

    if writer_result:
        print("\n--- WriterAgent 结果 ---")
        print(f"来源: {writer_result.source_agent}")
        print(f"内容 ({len(writer_result.content)} 字符):")
        print(writer_result.content)
    else:
        print(f"\nWriterAgent 未能为章节 '{test_section}' 生成结果")