from src.actions.base_action import BaseAction
from src.schemas import Task, Result
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CallAnalysisAgentAction(BaseAction):
    """负责调用 AnalysisAgent 的 Action"""
    
    def __init__(self, analysis_agent):
        self.analysis_agent = analysis_agent
    
    @property
    def name(self) -> str:
        return "CallAnalysisAgent"
    
    @property
    def description(self) -> str:
        return "调用 AnalysisAgent 分析指定章节的研究结果"
    
    async def run(self, research_result: Result, section_topic: str, original_query: str, language: str) -> Dict[str, Any]:
        """
        调用 AnalysisAgent 分析研究结果
        
        Args:
            research_result: ResearchAgent 的结果
            section_topic: 章节主题
            original_query: 原始查询
            language: 目标语言
            
        Returns:
            包含分析结果的字典
        """
        logger.info(f"调用 AnalysisAgent 分析章节 '{section_topic}' 的研究结果")
        
        # 创建用于 AnalysisAgent 的任务对象
        task = Task(query=original_query, language=language)
        
        result = await self.analysis_agent.run(task, research_result, section_topic, language)
        
        if not result or not result.content:
            logger.error(f"AnalysisAgent 未返回有效结果")
            return {"success": False, "error": "分析失败", "section_topic": section_topic, "result": None}
        
        # 检查分析结果是否充分
        content_dict = result.content
        is_sufficient = content_dict.get('sufficient', False) if isinstance(content_dict, dict) else False
        
        logger.info(f"AnalysisAgent 成功返回章节 '{section_topic}' 的分析结果 (充分: {is_sufficient})")
        return {
            "success": True,
            "section_topic": section_topic,
            "result": result,
            "sufficient": is_sufficient
        }