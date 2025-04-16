import logging
from typing import Any, Dict, List

from .base_action import BaseAction
# 导入 LLM 客户端基类用于类型提示
from ..core.llm_integration import BaseLLMClient
# 不再需要导入 generate_text
# from ..core.llm_integration import generate_text

# Setup logger for this action
logger = logging.getLogger(__name__)
# 移除这里的 basicConfig，假设日志在应用程序入口处配置
# logging.basicConfig(level=logging.INFO)

class CreatePlanAction(BaseAction):
    """
    一个 Action，用于根据用户请求调用 LLM 生成报告大纲（计划）。
    """

    @property
    def name(self) -> str:
        return "create_plan"

    @property
    def description(self) -> str:
        return "Generates a structured report outline (list of section titles) based on the user request provided as input."

    # 更新 run 方法签名以接收 llm_client
    def run(self, action_input: Any, llm_client: BaseLLMClient, **kwargs) -> Dict[str, Any]:
        """
        使用 LLM 生成报告大纲。

        Args:
            action_input: 用户的原始请求 (query)。
            llm_client: LLM 客户端实例。 # <--- 现在从这里接收
            **kwargs: 未使用。

        Returns:
            包含生成状态和结果 (大纲列表) 的字典。
        """
        if not isinstance(action_input, str) or not action_input:
            logger.warning("Action 'create_plan' received invalid input. Expected a non-empty string.")
            # 保持一致的返回结构
            return {"status": "Error", "error": "Invalid input for create_plan. Expected the user query.", "outline": []}

        query = action_input
        logger.info(f"Executing 'create_plan' action for query: '{query}'")

        # 不再需要检查 llm_client 是否为 None，因为 BaseAgent 会确保传入有效的实例
        # if llm_client is None: ... (Removed)

        # 构建调用 LLM 生成大纲的 Prompt
        outline_prompt = f"""Please generate a detailed list of section titles for a report about: "{query}"

Provide the output ONLY as a numbered or bulleted list of titles, each on a new line. Do not include any introductory text, explanations, or concluding remarks. Just the list.

Example:
1. Introduction
2. Key Findings
3. Analysis
4. Conclusion
"""
        try:
            # --- 使用传入的 llm_client 实例调用 LLM ---
            logger.debug(f"Sending prompt to LLM for outline generation:\n{outline_prompt}")
            # 调用客户端实例的 generate 方法
            response = llm_client.generate(outline_prompt)
            logger.debug(f"Received response from LLM:\n{response}")
            # --- ---------------------------------- ---

            # 检查 LLM 客户端是否返回了错误信息
            if response is None: # 处理 generate 可能返回 None 的情况 (尽管我们尝试避免)
                 logger.error("LLM client returned None unexpectedly.")
                 return {"status": "Error", "error": "LLM client returned None.", "outline": []}
            elif response.startswith("Error:"):
                logger.error(f"LLM generation failed: {response}")
                # 将 LLM 返回的错误信息传递出去
                return {"status": "Error", "error": response, "outline": []}

            # 解析 LLM 返回的大纲文本为列表 (保持之前的解析逻辑)
            lines = response.strip().split('\n')
            outline = []
            for line in lines:
                line = line.strip()
                if line:
                    if line[0].isdigit():
                        parts = line.split('.', 1)
                        if len(parts) == 2:
                            title = parts[1].strip()
                            if title: outline.append(title)
                    elif line.startswith(('-', '*')):
                         title = line[1:].strip()
                         if title: outline.append(title)
                    else:
                         title = line.strip()
                         if title and not title.endswith(':'): # 简单的过滤
                            outline.append(title)

            if not outline:
                 logger.warning(f"Could not parse any outline items from LLM response for query '{query}'. Response: {response}")
                 # 返回部分原始响应作为提示，标记为警告
                 return {"status": "Warning", "message": "Failed to parse outline from LLM response.", "outline": [f"Raw response snippet: {response[:100]}..."]}
            else:
                logger.info(f"Successfully generated outline with {len(outline)} sections for query: '{query}'.")
                # 成功时返回 status: Success
                return {"status": "Success", "message": f"Successfully generated outline with {len(outline)} sections.", "outline": outline}

        except Exception as e:
            logger.error(f"Error during 'create_plan' action execution: {e}", exc_info=True)
            return {"status": "Error", "error": f"Exception during action execution: {e}", "outline": []}
