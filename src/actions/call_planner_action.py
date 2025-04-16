from src.actions.base_action import BaseAction
from src.schemas import Task, Result
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CallPlannerAgentAction(BaseAction):
    """负责调用 PlannerAgent 的 Action"""
    
    def __init__(self, planner_agent):
        self.planner_agent = planner_agent
    
    @property
    def name(self) -> str:
        return "CallPlannerAgent"
    
    @property
    def description(self) -> str:
        return "调用 PlannerAgent 将用户任务分解为报告的章节大纲"
    
    async def run(self, task: Task) -> Dict[str, Any]:
        """
        调用 PlannerAgent 生成报告计划
        
        Args:
            task: 原始用户任务
            
        Returns:
            包含计划信息的字典，至少包含 sections (章节列表)
        """
        logger.info(f"调用 PlannerAgent 对任务 '{task.query}' 进行规划")
        
        # PlannerAgent.run不是异步方法，直接调用
        result = self.planner_agent.run(user_query=task.query)
        
        if not result or result.get("status") != "Success":
            logger.error("PlannerAgent 未返回有效结果")
            return {"success": False, "error": "规划失败", "sections": []}
        
        # 从planner结果中提取章节信息
        if "history" in result:
            # 查找包含成功大纲的步骤
            sections = []
            for step in reversed(result['history']):
                if step.get("action") == "create_plan" and isinstance(step.get("observation"), dict) and "outline" in step["observation"]:
                    sections = step["observation"]["outline"]
                    break
            
            if sections:
                logger.info(f"PlannerAgent 成功生成计划，包含 {len(sections)} 个章节")
                return {"success": True, "sections": sections, "full_plan": {"sections": sections}}
        
        logger.error(f"PlannerAgent 返回的结果格式不正确: {result}")
        return {"success": False, "error": "规划结果格式错误", "sections": []}