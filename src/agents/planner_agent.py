import logging
from typing import List, Dict, Any, Optional

from .base_agent import BaseAgent
# 显式导入 Action 和 LLMClient 类型
from ..actions.base_action import BaseAction
from ..core.llm_integration import BaseLLMClient

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    规划 Agent，负责理解用户需求，并使用注入的 'create_plan' Action 生成报告大纲。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient, # <--- 修改：接收 BaseLLMClient
        actions: List[BaseAction], # <--- 明确类型为 BaseAction
        agent_name: str = "PlannerAgent",
        task_description: str = "Generate a report outline (section titles) based on the user request using the 'create_plan' action.",
        max_iterations: int = 2,
        verbose: bool = False,
        # state_manager: Optional[Any] = None
    ):
        # 验证 'create_plan' Action 是否存在
        if not any(action.name == "create_plan" for action in actions):
             raise ValueError("PlannerAgent requires 'create_plan' action in the actions list.")

        # 将注入的 llm_client 和 actions 传递给基类
        super().__init__(
            llm_client=llm_client, # <--- 传递 llm_client
            actions=actions,
            agent_name=agent_name,
            task_description=task_description,
            max_iterations=max_iterations,
            verbose=verbose,
            # state_manager=state_manager
        )

    def _build_prompt(self, intermediate_steps: List[Dict[str, Any]], user_query: str) -> str:
        """
        构建 Planner Agent 的 Prompt。
        指示 LLM 基于用户请求生成报告大纲，并使用 create_plan action 输出。

        Args:
            intermediate_steps: 历史步骤。
            user_query: 用户的原始请求。

        Returns:
            构建好的 Prompt 字符串。
        """
        action_descriptions = self._get_available_action_descriptions() # 从基类获取描述

        prompt = f"""You are a Planner Agent responsible for creating a structured report outline.
Your primary goal is to generate a list of relevant section titles for a report based on the following user request:
"{user_query}"

Available Actions:
{action_descriptions}

Think step-by-step about what sections would be appropriate for this report.
Then, use the 'create_plan' action to output the generated outline. The input for 'create_plan' should be the original user request.

History of steps taken so far:
"""
        if not intermediate_steps:
            prompt += "No steps taken yet. Your first step should be to use 'create_plan'.\n"
        else:
            for step in intermediate_steps:
                prompt += f"Iteration: {step.get('iteration', 'N/A')}\n"
                prompt += f"Thought: {step.get('thought', '')}\n"
                prompt += f"Action: {step.get('action', '')}\n"
                prompt += f"Action Input: {step.get('action_input', '')}\n"
                # 格式化 Observation 以便 LLM 理解
                obs = step.get('observation')
                if isinstance(obs, dict):
                     # 只显示关键信息，避免过多细节淹没 LLM
                     obs_str = f"Status: {obs.get('status', 'N/A')}, Message: {obs.get('message', obs.get('error', ''))}"
                     if "outline" in obs:
                         obs_str += f", Outline Snippet: {str(obs['outline'][:2]) + '...' if len(obs['outline']) > 2 else str(obs['outline'])}" # 显示部分大纲
                else:
                     obs_str = str(obs)
                prompt += f"Observation: {obs_str}\n\n"

            # 检查上一步是否成功生成了计划
            last_observation = intermediate_steps[-1].get("observation")
            if isinstance(last_observation, dict) and last_observation.get("status") == "Success" and "outline" in last_observation:
                 prompt += "The outline has been successfully generated in the last step. Your task is complete. Use the 'final_answer' action to return the successful result message.\n"
            elif isinstance(last_observation, dict) and last_observation.get("status") == "Error":
                 prompt += "The last action resulted in an error. Report this error using the 'final_answer' action.\n"
            else:
                 prompt += "Review the history. If the outline is not yet generated or failed previously, use 'create_plan'.\n"


        prompt += f"""
Based on the user request "{user_query}" and the history, determine the next step. Output your thought process and the action in the specified format:

Thought: [Your reasoning]
Action: [Action name, e.g., create_plan or final_answer]
Action Input: [Input for the action]
"""
        # 不再硬编码 Action，让 LLM 根据 Prompt 和历史决定
        return prompt

    # 不再需要 _generate_outline_with_llm 方法，逻辑移到 CreatePlanAction
    # 不再需要覆盖 _execute_action 方法，基类会处理 Action 调用

    # run 方法可以简化，直接使用基类的，或者保留用于类型提示
    def run(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """
        运行 Planner Agent。

        Args:
            user_query: 用户的原始请求。
            **kwargs: 传递给基类 run 的其他参数。

        Returns:
            一个包含最终结果和执行历史的字典。
        """
        logger.info(f"PlannerAgent starting run with user query: '{user_query}'")
        # 将 user_query 作为关键参数传递给基类的 run，基类 run 会再传递给 _build_prompt
        result = super().run(user_query=user_query, **kwargs)
        logger.info(f"PlannerAgent finished run.")
        return result


# 第五步：更新示例用法
if __name__ == '__main__':
    # 确保日志已配置 (例如，在 main.py 或此处临时配置)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- 设置依赖 ---
    try:
        # 1. 从 core 导入 Action 和 LLM 获取函数
        from src.actions.create_plan_action import CreatePlanAction
        from src.core.llm_integration import get_llm_client, BaseLLMClient # 导入工厂和基类
        # (可选) 导入日志设置
        # from src.utils.log_setup import setup_logging
        # setup_logging(log_level=logging.DEBUG) # 设置更详细日志

    except ImportError as e:
        logger.error(f"Failed to import necessary modules. Make sure PYTHONPATH is set correctly. Error: {e}")
        import sys
        sys.exit(1)


    # --- 实例化依赖 ---
    try:
        # 2. 获取 LLM 客户端实例 (依赖 .env 文件配置)
        llm_client: BaseLLMClient = get_llm_client()
        logger.info(f"Successfully obtained LLM client: {type(llm_client).__name__}")

        # 3. 实例化需要的 Action
        create_plan_action = CreatePlanAction()
        actions_list = [create_plan_action] # 可以添加其他 Planner 需要的 Action

    except (ValueError, ImportError, Exception) as e:
        logger.error(f"Failed to initialize dependencies (LLM Client or Actions): {e}", exc_info=True)
        import sys
        sys.exit(1)

    # --- 实例化并运行 Agent ---
    try:
        # 4. 实例化 Agent，注入 LLM 客户端和 Actions
        planner = PlannerAgent(
            llm_client=llm_client,
            actions=actions_list,
            verbose=True # 开启 BaseAgent 的详细日志 (思考过程)
        )

        # 5. 运行 Agent
        user_request = "生成一份关于 AI 行业最新发展趋势报告,只需要包含3个章节"
        final_result = planner.run(user_query=user_request)

        # 6. 处理结果
        print("\n\n===================================")
        print("=== Planner Agent Final Result ===")
        print("===================================")
        import json
        # 打印整个结果字典，其中包含 history
        print(json.dumps(final_result, indent=2, ensure_ascii=False))

        # 方便地提取最终大纲（如果成功）
        if final_result.get("status") == "Success" and final_result.get("history"):
             last_step = final_result['history'][-1]
             # 查找包含成功大纲的步骤 (可能是 final_answer 之前的 create_plan 步骤)
             outline_step = None
             for step in reversed(final_result['history']):
                 if step.get("action") == "create_plan" and isinstance(step.get("observation"), dict) and step["observation"].get("status") == "Success":
                     outline_step = step
                     break
             if outline_step and "outline" in outline_step["observation"]:
                 print("\n--- Generated Outline ---")
                 print(json.dumps(outline_step["observation"]["outline"], indent=2, ensure_ascii=False))

    except Exception as e:
         logger.exception("An error occurred during PlannerAgent execution.", exc_info=True)
