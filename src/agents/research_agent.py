import logging
from typing import Optional, List, Dict, Any
import ast # 导入 ast 模块用于安全地解析字符串

from .base_agent import BaseAgent
from ..core.llm_integration import BaseLLMClient
from ..actions.base_action import BaseAction
from ..core.state_manager import StateManager  # 导入状态管理器

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """
    研究 Agent，负责根据 PlannerAgent 生成的大纲收集相关资料。
    使用 search 和 browse 两个 Action 完成任务。
    采用 ReAct（思考-行动-观察）模式实现。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        actions: List[BaseAction],
        agent_name: str = "ResearchAgent",
        task_description: str = "收集报告章节所需的相关信息，通过搜索和浏览网页内容来获取资料。",
        max_iterations: int = 10,  # 减少迭代次数，加快测试速度
        verbose: bool = False,
        state_manager: Optional[StateManager] = None
    ):
        # 验证是否传入了必要的 Action
        required_actions = ["search", "browse"]
        available_actions = [action.name for action in actions]
        missing_actions = [name for name in required_actions if name not in available_actions]
        
        if missing_actions:
            raise ValueError(f"ResearchAgent requires the following actions: {missing_actions}")

        super().__init__(
            llm_client=llm_client,
            actions=actions,
            agent_name=agent_name,
            task_description=task_description,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        
        # 初始化或使用传入的状态管理器
        self.state_manager = state_manager if state_manager else StateManager()
        logger.info(f"ResearchAgent initialized with max_iterations: {max_iterations}")

    def _build_prompt(self, outline: List[str], current_section_index: int, iterations: List[Dict] = None, **kwargs) -> str:
        """
        构建研究代理的提示，使用状态管理器获取当前状态。
        """
        current_section = outline[current_section_index] if 0 <= current_section_index < len(outline) else "未知章节"
        
        # 确保状态管理器知道当前章节
        if self.state_manager.current_section != current_section:
            self.state_manager.set_current_section(current_section)
            logger.debug(f"State Manager current section updated to: {current_section}")

        # 获取状态摘要，用于构建prompt
        state_summary = self.state_manager.get_state_summary()
        recommended_action = self.state_manager.get_recommended_action()
        logger.debug(f"Building prompt with state summary: {state_summary}")
        logger.debug(f"Recommended action: {recommended_action}")

        # 基本提示信息
        prompt = f"""你是一个专业的研究助手，负责为"{current_section}"章节收集和整理信息。

你将使用以下工具完成任务：
1. search - 在网络上搜索信息。输入：搜索查询字符串。
2. browse - 获取特定URL的网页内容。输入：URL字符串或URL列表（如 ["url1", "url2", ...]）。
3. final_answer - 提供最终的研究结果。输入：基于收集到的信息撰写的完整章节内容。

工作流程：
1. 你应该先用search工具获取相关内容的搜索结果
2. 然后分析搜索结果，选择最相关的网页，使用browse工具获取详细内容
3. 分析获取的内容，确认是否需要进一步搜索更多信息
4. 当收集到足够的信息后，使用final_answer提交完整的研究结果

重要提示：
- 对于browse工具，你可以同时提供多个URL以一次获取多个页面的内容，格式为URL列表: ["url1", "url2", ...]
- 确保在最终答案中综合多个来源的信息，并提供结构化的内容
- 最终答案应全面涵盖章节主题，内容丰富有价值
- 请严格遵循以下格式：每步必须包含"思考"、"动作"和"动作输入"三部分

--- 当前研究状态摘要 ---
当前章节: "{state_summary['current_section']}"
当前阶段: {state_summary['stage']}
已执行搜索次数: {state_summary['searches_performed']}
已浏览URL数: {state_summary['urls_browsed']}
待浏览URL数: {state_summary['pending_urls_count']}
"""

        # 添加最近的搜索信息
        if 'latest_search' in state_summary:
            latest_search = state_summary['latest_search']
            prompt += f"\n最近的搜索查询: \"{latest_search['query']}\""
            prompt += f"\n找到结果数量: {latest_search['results_count']}"
            
            if 'top_results' in latest_search and latest_search['top_results']:
                prompt += "\n部分搜索结果:"
                for i, result in enumerate(latest_search['top_results']):
                    prompt += f"\n  {i+1}. {result['title']}"
                    prompt += f"\n     URL: {result['url']}"
                    prompt += f"\n     摘要: {result['snippet']}"
        
        # 添加待浏览URL信息
        if 'pending_urls' in state_summary and state_summary['pending_urls']:
            prompt += "\n\n待浏览的URL:"
            for i, url_info in enumerate(state_summary['pending_urls']):
                prompt += f"\n  {i+1}. {url_info['title']}"
                prompt += f"\n     URL: {url_info['url']}"
        
        # 添加已浏览内容摘要
        if 'browsed_content_summaries' in state_summary and state_summary['browsed_content_summaries']:
            prompt += "\n\n已浏览内容摘要:"
            for i, content in enumerate(state_summary['browsed_content_summaries']):
                prompt += f"\n  {i+1}. URL: {content['url']}"
                prompt += f"\n     摘要: {content['summary']}"
        
        # 添加系统建议的下一步行动
        prompt += f"\n\n--- 系统建议的下一步行动 ---"
        prompt += f"\n建议执行: {recommended_action['action']}"
        prompt += f"\n原因: {recommended_action['explanation']}"
        
        if recommended_action['action'] == 'browse' and 'urls' in recommended_action:
            prompt += "\n建议浏览的URL:"
            for i, url in enumerate(recommended_action['urls']):
                prompt += f"\n  {i+1}. {url}"
        
        # 最终决策指南
        prompt += f"""

--- 下一步行动决策 ---
请根据以上状态摘要和建议，为章节 "{current_section}" 的研究确定你的下一步行动。
请注意以下几点:
1. 如果尚未执行搜索或需要获取更多相关信息，使用 search 操作
2. 如果有待浏览的URL且信息看起来相关，使用 browse 操作（可以同时浏览多个URL）
3. 如果已经收集到足够的信息，使用 final_answer 操作提供全面的研究结果

你的决策应包含：
思考: [分析当前状态，并解释你为什么选择下一步行动]
动作: [选择 search、browse 或 final_answer]
动作输入: [根据动作提供相应的输入：搜索查询、URL或URL列表、或最终答案]
"""
        return prompt

    def _process_observation(self, action: str, action_input: Any, observation: Dict[str, Any]) -> None:
        """
        处理Action执行后的观察结果，更新状态管理器。
        """
        logger.debug(f"Processing observation for action: {action}")
        if action == "search" and observation.get("status") == "Success" and "results" in observation:
            # 处理搜索结果
            self.state_manager.add_search_result(action_input, observation["results"])
            logger.debug(f"Updated state manager with search results for query: '{action_input}'")
        
        elif action == "browse" and observation.get("status") == "Success" and "content" in observation:
            # 处理浏览结果
            urls = action_input if isinstance(action_input, list) else [action_input]
            self.state_manager.add_browsed_content(urls, observation["content"])
            logger.debug(f"Updated state manager with browsed content for {len(urls)} URLs")
        
        elif action == "final_answer":
            # 处理最终答案
            self.state_manager.set_final_answer(action_input)
            logger.debug("Updated state manager with final answer")
        else:
            logger.debug(f"No state update needed for action '{action}' or observation status is not Success.")

    def run(self, outline: List[str], **kwargs) -> Dict[str, Any]:
        """
        为报告大纲的每个章节收集资料。
        """
        logger.info(f"ResearchAgent starting to gather information for {len(outline)} sections.")
        
        # 存储每个章节的研究结果
        section_results = {}
        
        # 对每个章节进行研究
        for i, section in enumerate(outline):
            logger.info(f"--- Researching section {i+1}/{len(outline)}: '{section}' ---")
            
            # 重置状态管理器，准备处理新章节
            self.state_manager.reset()
            self.state_manager.set_current_section(section)
            logger.info(f"StateManager reset for section: '{section}'")
            
            # 为当前章节运行 ReAct 循环 (调用 BaseAgent 的 run)
            logger.info(f"Calling BaseAgent.run for section: '{section}'")
            try:
                # 假设 BaseAgent.run 处理单个任务的ReAct循环
                # 它会使用 self._build_prompt 和 self._execute_action (被重写的版本)
                base_run_result = super().run(
                    outline=outline,
                    current_section_index=i,
                    **kwargs
                )

                # 提取最终结果，可能来自 base_run_result 或 state_manager
                final_answer = self.state_manager.final_answer
                if not final_answer:
                     final_answer = base_run_result.get("final_answer", "Research stopped before final answer.")
                
                status = base_run_result.get("status", "Unknown") # Get status from BaseAgent.run result
                if status == "Success" and self.state_manager.stage != "finalizing":
                     status = "Completed without final answer" # More specific status
                
                section_result = {
                    "status": status,
                    "message": base_run_result.get("message", ""),
                    "final_answer": final_answer,
                    "history": base_run_result.get("history", [])
                }
                logger.info(f"BaseAgent.run completed for section: '{section}' with status: {status}")

            except Exception as e:
                logger.error(f"Error during BaseAgent.run for section '{section}': {e}", exc_info=True)
                section_result = {
                    "status": "Error",
                    "message": f"Failed to complete research for section due to error: {str(e)}",
                    "final_answer": f"无法完成对 '{section}' 的研究，发生错误：{str(e)}"
                }
            
            # 存储结果
            section_results[section] = section_result
            logger.info(f"--- Completed research for section: '{section}' ---")
        
        logger.info(f"ResearchAgent finished gathering information for all sections.")
        return {
            "status": "Success",
            "message": f"Completed research for {len(outline)} sections.",
            "section_results": section_results
        }

    # Override _execute_action with input parsing for browse
    def _execute_action(self, action: str, action_input: Any) -> Dict[str, Any]:
        """
        执行动作并更新状态管理器。
        增加了对 browse 动作输入的特殊处理，以解析字符串形式的列表。
        """
        logger.debug(f"Overridden _execute_action called for action: {action}")
        
        parsed_action_input = action_input # 默认使用原始输入

        # === 新增解析逻辑开始 ===
        if action == "browse" and isinstance(action_input, str):
            logger.debug(f"Attempting to parse string action_input for browse: {action_input[:100]}...") # Log beginning of string
            try:
                # 尝试将字符串解析为 Python 列表
                # ast.literal_eval 比 eval 更安全
                parsed_input = ast.literal_eval(action_input)
                if isinstance(parsed_input, list):
                    # 确保列表中的元素是字符串
                    if all(isinstance(item, str) for item in parsed_input):
                         parsed_action_input = parsed_input
                         logger.debug(f"Successfully parsed action_input into list: {parsed_action_input}")
                    else:
                         logger.warning(f"Parsed list contains non-string elements: {parsed_input}. Falling back to original string.")
                else:
                    logger.warning(f"Parsed action_input is not a list: {type(parsed_input)}. Falling back to original string.")
            except (ValueError, SyntaxError, TypeError) as e:
                logger.warning(f"Could not parse action_input string as list: {e}. Falling back to original string.")
        # === 新增解析逻辑结束 ===

        # 使用解析后的输入调用父类或直接执行动作
        try:
             # 传递解析后的输入 (parsed_action_input)
             observation = super()._execute_action(action, parsed_action_input)
             logger.debug(f"BaseAgent._execute_action completed for action: {action}")
        except AttributeError:
             logger.warning(f"BaseAgent does not have _execute_action. Falling back to direct action execution.")
             action_object = self.action_map.get(action)
             if not action_object:
                 logger.error(f"Action '{action}' not found.")
                 return {"status": "Error", "error": f"Action '{action}' not found."}
             try:
                 # 传递解析后的输入 (parsed_action_input)
                 observation = action_object.run(parsed_action_input, self.llm_client)
             except Exception as e:
                 logger.error(f"Exception during action '{action}': {e}", exc_info=True)
                 observation = {"status": "Error", "error": f"Exception during action '{action}': {e}"}

        # 然后更新状态管理器 (使用原始 action_input 来记录，或者解析后的？保持原始可能更好追踪)
        # Let's use original action_input for add_search_result/add_browsed_content consistency
        self._process_observation(action, action_input, observation)

        return observation

if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 导入必要的模块
    try:
        from src.actions.search_action import SearchAction
        from src.actions.browse_action import BrowseAction
        from src.core.llm_integration import get_llm_client
        from src.core.state_manager import StateManager
    except ImportError:
        # 如果无法使用相对导入，尝试使用绝对导入
        from ..actions.search_action import SearchAction
        from ..actions.browse_action import BrowseAction
        from ..core.llm_integration import get_llm_client
        from ..core.state_manager import StateManager
    
    # 获取 LLM 客户端
    llm_client = get_llm_client()
    logger.info("LLM client obtained, initializing actions...")
    
    # 实例化 Action
    search_action = SearchAction()
    browse_action = BrowseAction()
    
    # 实例化状态管理器
    state_manager = StateManager()
    
    # 实例化 ResearchAgent
    researcher = ResearchAgent(
        llm_client=llm_client,
        actions=[search_action, browse_action],
        verbose=True,
        max_iterations=3,  # 增加迭代次数以获得更完整的研究结果
        state_manager=state_manager
    )
    logger.info("ResearchAgent initialized, starting test run...")
    
    # 测试大纲
    test_outline = [
        "人工智能的发展历程"
    ]
    
    # 运行研究
    try:
        logger.info("Running research with test outline...")
        results = researcher.run(outline=test_outline)
        logger.info("Research completed successfully")
    except Exception as e:
        logger.error(f"Error running research: {e}", exc_info=True)
        results = {"status": "Error", "message": f"Research failed: {str(e)}"}
    
    # 打印结果摘要
    print("\n=== Research Results Summary ===")
    for section, result in results.get("section_results", {}).items():
        print(f"\nSection: {section}")
        status = result.get("status", "Unknown")
        print(f"Status: {status}")
        if status == "Success" and "final_answer" in result:
            print(f"Summary length: {len(result['final_answer'])} characters")
            print(f"Summary: {result['final_answer'][:300]}...")  # 仅打印开头部分
