from src.actions.base_action import BaseAction
from src.schemas import Task, Result
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CallWriterAgentAction(BaseAction):
    """负责调用 WriterAgent 的 Action"""
    
    def __init__(self, writer_agent):
        self.writer_agent = writer_agent
    
    @property
    def name(self) -> str:
        return "CallWriterAgent"
    
    @property
    def description(self) -> str:
        return "调用 WriterAgent 根据分析结果撰写指定章节的文本"
    
    async def run(self, analysis_result: Result, section_topic: str, original_query: str, language: str) -> Dict[str, Any]:
        """
        调用 WriterAgent 撰写章节文本
        
        Args:
            analysis_result: AnalysisAgent 的结果
            section_topic: 章节主题
            original_query: 原始查询
            language: 目标语言
            
        Returns:
            包含撰写结果的字典
        """
        logger.info(f"调用 WriterAgent 撰写章节 '{section_topic}' 的文本")
        
        # 创建用于 WriterAgent 的任务对象
        task = Task(query=original_query, language=language)
        
        result = await self.writer_agent.run(task, analysis_result, section_topic, language)
        
        if not result or not result.content:
            logger.error(f"WriterAgent 未返回有效结果")
            return {"success": False, "error": "撰写失败", "section_topic": section_topic, "result": None}
        
        logger.info(f"WriterAgent 成功返回章节 '{section_topic}' 的文本 ({len(result.content)} 字符)")
        return {
            "success": True,
            "section_topic": section_topic,
            "result": result
        }