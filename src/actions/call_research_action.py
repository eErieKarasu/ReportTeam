from src.actions.base_action import BaseAction
from src.schemas import Task, Result
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CallResearchAgentAction(BaseAction):
    """负责调用 ResearchAgent 的 Action"""
    
    def __init__(self, research_agent):
        self.research_agent = research_agent
    
    @property
    def name(self) -> str:
        return "CallResearchAgent"
    
    @property
    def description(self) -> str:
        return "调用 ResearchAgent 为指定章节搜集信息"
    
    async def run(self, section_topic: str, original_query: str, language: str) -> Dict[str, Any]:
        """
        为指定章节调用 ResearchAgent 搜集信息
        
        Args:
            section_topic: 章节主题
            original_query: 原始查询
            language: 目标语言
            
        Returns:
            包含研究结果的字典
        """
        logger.info(f"调用 ResearchAgent 为章节 '{section_topic}' 搜集信息")
        
        # ResearchAgent.run接受outline参数，而不是task
        outline = [section_topic]
        
        # 直接调用非异步的research_agent.run方法
        result = self.research_agent.run(outline=outline)
        
        if not result or result.get("status") != "Success":
            logger.error(f"ResearchAgent 未返回有效结果")
            return {"success": False, "error": "研究失败", "section_topic": section_topic, "result": None}
        
        # 从结果中提取当前章节的结果
        section_results = result.get("section_results", {})
        if section_topic not in section_results:
            logger.error(f"ResearchAgent 未返回章节 '{section_topic}' 的结果")
            return {"success": False, "error": f"未找到章节 '{section_topic}' 的研究结果", "section_topic": section_topic, "result": None}
        
        section_result = section_results[section_topic]
        if section_result.get("status") != "Success":
            logger.error(f"ResearchAgent 返回章节 '{section_topic}' 的结果失败: {section_result.get('message', '未知错误')}")
            return {"success": False, "error": section_result.get('message', "章节研究失败"), "section_topic": section_topic, "result": None}
        
        # 创建结果对象
        research_content = section_result.get("final_answer", "")
        research_result = Result(content=research_content, source_agent="ResearchAgent")
        
        logger.info(f"ResearchAgent 成功返回章节 '{section_topic}' 的研究结果")
        return {
            "success": True,
            "section_topic": section_topic,
            "result": research_result
        }