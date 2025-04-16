from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# 导入 LLM 客户端基类和 Action 基类 (用于类型提示)
from ..core.llm_integration import BaseLLMClient
from ..actions.base_action import BaseAction # 使用 Any 仍然可以，但 BaseAction 更明确

# 添加日志记录
import logging
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Agent 的抽象基类，定义了基于 ReAct 模式的核心逻辑框架。
    使用依赖注入的 LLM 客户端和 Action 对象。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient, # <--- 修改：接收 BaseLLMClient 实例
        actions: List[BaseAction], # 使用 BaseAction 类型提示
        agent_name: str,
        task_description: str,
        max_iterations: int = 5,
        verbose: bool = False,
        # state_manager: Optional[Any] = None
    ):
        """
        初始化 Agent。

        Args:
            llm_client: 实现 BaseLLMClient 接口的 LLM 客户端实例。
            actions: Agent 可用的 Action 对象列表 (实现 BaseAction 接口)。
            agent_name: Agent 的名称。
            task_description: Agent 需要完成的任务描述。
            max_iterations: ReAct 循环的最大迭代次数。
            verbose: 是否打印详细过程。
            state_manager: 状态管理器 (可选)。
        """
        # 移除之前的检查，因为类型提示已确保 llm_client 不为 None
        # if llm_client is None:
        #     raise ValueError("llm_client cannot be None")

        self.llm_client = llm_client # <--- 存储注入的客户端
        self.actions = {action.name: action for action in actions}
        self.agent_name = agent_name
        self.task_description = task_description
        self.max_iterations = max_iterations
        self.verbose = verbose
        # self.state_manager = state_manager

        self.intermediate_steps: List[Dict[str, Any]] = []

        logger.info(f"Initializing Agent: {self.agent_name}")
        logger.info(f"LLM Client Type: {type(self.llm_client).__name__}")
        logger.info(f"Available Actions: {list(self.actions.keys())}")


    @abstractmethod
    def _build_prompt(self, intermediate_steps: List[Dict[str, Any]], **kwargs) -> str:
        """
        构建发送给 LLM 的 Prompt。
        子类需要实现此方法，并应包含可用 Action 的描述。
        """
        pass

    def _get_available_action_descriptions(self) -> str:
        """Helper to get formatted descriptions of available actions for the prompt."""
        descriptions = []
        for name, action in self.actions.items():
            descriptions.append(f"- {name}: {action.description}")
        return "\n".join(descriptions) if descriptions else "No actions available."


    def _think(self, prompt: str) -> str:
        """
        调用注入的 LLM 客户端进行思考，生成下一步的行动计划。
        """
        logger.info(f"--- {self.agent_name} Thinking ---")
        if self.verbose:
            # 使用 logger.debug 输出详细信息
            logger.debug(f"Sending Prompt to LLM ({type(self.llm_client).__name__}):\n-------\n{prompt}\n-------")

        # --- 调用注入的 LLM 客户端 ---
        try:
            # 使用 self.llm_client
            response = self.llm_client.generate(prompt)
            # 检查是否返回 None (虽然我们尝试避免)
            if response is None:
                 logger.error("LLM client generate method returned None.")
                 return "Error: LLM client returned None."
            logger.debug(f"Received Raw Response from LLM:\n-------\n{response}\n-------")
            return response
        except Exception as e:
            logger.error(f"Error during LLM call in _think using {type(self.llm_client).__name__}: {e}", exc_info=True)
            # 返回错误信息，让解析步骤可以处理
            return f"Error: Failed to get response from LLM: {e}"
        # --- ----------------------- ---


    def _parse_llm_output(self, llm_output: str) -> Dict[str, str]:
        """
        解析 LLM 的输出，提取出行动指令（Action 名称和输入）。
        支持中英文格式的输出解析。
        """
        logger.debug(f"Parsing LLM Output:\n{llm_output}")
        if llm_output.startswith("Error:"): # 处理 LLM 调用或返回 None 的错误
             logger.error(f"LLM call failed or returned error: {llm_output}")
             # 返回 thought, action, action_input 结构
             return {"thought": f"Error occurred: {llm_output}", "action": "error", "action_input": llm_output}

        thought = ""
        action = "error" # Default to error if parsing fails
        action_input = "Could not parse action from LLM output."

        try:
            # 定义中英文关键词
            thought_keywords = ["Thought:", "思考:"]
            action_keywords = ["Action:", "动作:"]
            action_input_keywords = ["Action Input:", "动作输入:"]
            final_answer_keywords = ["Final Answer:", "最终答案:"]
            
            # 识别使用的语言格式
            has_thought = False
            has_action = False
            used_thought_keyword = None
            used_action_keyword = None
            used_action_input_keyword = None
            
            # 检查使用的是哪种关键词
            for kw in thought_keywords:
                if kw in llm_output:
                    has_thought = True
                    used_thought_keyword = kw
                    break
                    
            for kw in action_keywords:
                if kw in llm_output:
                    has_action = True
                    used_action_keyword = kw
                    break
                    
            for kw in action_input_keywords:
                if kw in llm_output and has_action:
                    used_action_input_keyword = kw
                    break
            
            # 解析输出
            if has_thought and has_action:
                thought_part = llm_output.split(used_action_keyword, 1)[0]
                action_part = llm_output.split(used_action_keyword, 1)[1]
                
                # 提取思考内容
                thought = thought_part.replace(used_thought_keyword, "").strip()
                
                # 提取动作和动作输入
                if used_action_input_keyword and used_action_input_keyword in action_part:
                    action = action_part.split(used_action_input_keyword, 1)[0].strip()
                    action_input = action_part.split(used_action_input_keyword, 1)[1].strip()
                else:
                    action = action_part.strip()
                    action_input = ""
                    logger.warning(f"LLM output for action '{action}' did not contain explicit action input keyword. Assuming no input needed.")
            
            # 检查是否为最终答案
            elif any(kw in llm_output for kw in final_answer_keywords):
                # 确定使用的是哪个最终答案关键词
                used_final_kw = next((kw for kw in final_answer_keywords if kw in llm_output), final_answer_keywords[0])
                
                # 提取思考和最终答案
                if has_thought:
                    thought = llm_output.split(used_final_kw, 1)[0].replace(used_thought_keyword, "").strip()
                else:
                    thought = llm_output.split(used_final_kw, 1)[0].strip()
                    
                final_answer = llm_output.split(used_final_kw, 1)[1].strip()
                action = "final_answer"
                action_input = final_answer
            else:
                logger.warning(f"Could not parse standard Thought/Action/Input structure or Final Answer. Treating output as final answer/error.")
                thought = llm_output # Or try to extract thought if possible
                action = "final_answer" # Or 'error'?
                action_input = llm_output

        except Exception as e:
             logger.error(f"Error parsing LLM output: {e}\nOutput was:\n{llm_output}", exc_info=True)
             thought = "Error during parsing"
             action = "error"
             action_input = f"Parsing Error: {e}"

        parsed = {"thought": thought, "action": action, "action_input": action_input}
        logger.debug(f"Parsed LLM Output: {parsed}")
        return parsed


    def _execute_action(self, action_name: str, action_input: Any) -> Any:
        """
        执行指定的行动。查找对应的 Action 对象并调用其 run 方法，
        将 self.llm_client 传递给 Action。
        """
        logger.info(f"--- {self.agent_name} Executing Action ---")
        logger.info(f"Action Name: {action_name}")
        logger.info(f"Action Input: {action_input}")

        if action_name == "final_answer":
            logger.info("Action is 'final_answer'. Stopping.")
            return action_input
        elif action_name == "error":
             logger.error(f"Action is 'error'. Input: {action_input}. Stopping or handling error.")
             return {"status": "Error", "message": action_input}
        elif action_name in self.actions:
            action_object = self.actions[action_name]
            try:
                # --- 将 self.llm_client 传递给 Action 的 run 方法 ---
                # 其他需要的上下文也可以在这里传递，例如 state_manager
                observation = action_object.run(action_input, self.llm_client)
                # --- ------------------------------------------ ---
                logger.info(f"Observation from Action '{action_name}': {observation}")
                return observation
            except Exception as e:
                logger.error(f"Error executing action '{action_name}': {e}", exc_info=True)
                return {"status": "Error", "message": f"Exception during action '{action_name}': {e}"}
        else:
            logger.warning(f"Unknown action '{action_name}'. Cannot execute.")
            return {"status": "Error", "message": f"Unknown action '{action_name}' requested by LLM."}

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        运行代理的主要方法，实现ReAct模式的循环执行。

        Args:
            **kwargs: 传递给prompt构建函数的参数。

        Returns:
            包含执行结果的字典。
        """
        logger.info(f"\n=== Running Agent: {self.agent_name} ===")
        logger.info(f"Task: {self.task_description}")

        intermediate_steps = []
        final_result: Dict[str, Any] = {} # 初始化最终结果字典

        for i in range(self.max_iterations):
            logger.info(f"\n--- Iteration {i + 1} ---")
            step_log: Dict[str, Any] = {"iteration": i + 1} # 初始化当前步骤日志

            try:
                # 1. 构建 Prompt
                prompt = self._build_prompt(intermediate_steps=intermediate_steps, **kwargs)
                if prompt is None:
                     logger.error("Failed to build prompt. Stopping agent.")
                     final_result = {"status": "Error", "message": "Failed to build prompt."}
                     break # 退出循环

                # 2. 思考 (调用 LLM)
                llm_output = self._think(prompt)

                # 3. 解析行动
                parsed_output = self._parse_llm_output(llm_output)
                action = parsed_output["action"]
                action_input = parsed_output["action_input"]
                thought = parsed_output.get("thought", "")

                step_log.update({
                    "thought": thought,
                    "action": action,
                    "action_input": action_input
                })

                if action == "final_answer":
                    logger.info(f"--- {self.agent_name} Final Answer ---")
                    logger.info(action_input)
                    step_log["observation"] = action_input # 将最终答案记录为观察结果
                    intermediate_steps.append(step_log)
                    final_result = {"status": "Success", "final_answer": action_input, "history": intermediate_steps}
                    break # 退出循环
                elif action == "error":
                    logger.error(f"--- {self.agent_name} Encountered Error during parsing or LLM call ---")
                    logger.error(f"Error details: {action_input}")
                    step_log["observation"] = {"status": "Error", "message": action_input}
                    intermediate_steps.append(step_log)
                    final_result = {"status": "Error", "message": action_input, "history": intermediate_steps}
                    break # 退出循环

                # 4. 执行行动
                observation = self._execute_action(action, action_input)
                step_log["observation"] = observation
                intermediate_steps.append(step_log) # 在检查错误之前记录，以便看到失败的步骤

                # 检查 Action 执行是否出错
                if isinstance(observation, dict) and observation.get("status") == "Error":
                     logger.error(f"--- {self.agent_name} Action Execution Failed ---")
                     logger.error(f"Action '{action}' failed: {observation.get('message') or observation.get('error')}")
                     final_result = {"status": "Error", "message": f"Action '{action}' failed.", "details": observation, "history": intermediate_steps}
                     break # 退出循环

                 # (可选) 在这里添加提前结束的逻辑

            except Exception as e:
                logger.exception(f"Unexpected error during agent iteration {i+1}", exc_info=True)
                step_log["observation"] = {"status": "Error", "message": f"Unexpected agent error: {e}"}
                intermediate_steps.append(step_log) # 记录错误步骤
                final_result = {"status": "Error", "message": f"Unexpected agent error: {e}", "history": intermediate_steps}
                break # 退出循环

        # 处理循环正常结束但未设置 final_result 的情况 (通常是达到 max_iterations)
        if not final_result:
            logger.warning(f"--- {self.agent_name} Max Iterations ({self.max_iterations}) Reached ---")
            final_observation = intermediate_steps[-1].get("observation") if intermediate_steps else None
            final_result = {"status": "Max iterations reached", "final_observation": final_observation, "history": intermediate_steps}


        # 确保总是返回 history
        if "history" not in final_result:
             final_result["history"] = intermediate_steps

        return final_result
