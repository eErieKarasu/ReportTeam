from abc import ABC, abstractmethod
from typing import Any, Dict

# 导入我们新定义的 LLM 客户端基类
from ..core.llm_integration import BaseLLMClient

class BaseAction(ABC):
    """
    动作的抽象基类。
    每个动作代表 Agent 可以执行的一个具体操作。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """动作的唯一名称，用于 Agent 查找和调用。"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """动作的功能描述，用于 Agent (LLM) 理解其用途。"""
        pass

    @abstractmethod
    def run(self, action_input: Any, llm_client: BaseLLMClient, **kwargs) -> Dict[str, Any]:
        """
        执行动作的核心逻辑。

        Args:
            action_input: 从 Agent 传递过来的动作输入。
            llm_client: LLM 客户端实例，供需要调用 LLM 的动作使用。
            **kwargs: 其他可能需要的上下文信息（例如 state_manager）。

        Returns:
            一个包含执行结果的字典，至少应包含 'status' 或 'result'/'error' 键。
            Agent 会将此作为 Observation。
        """
        pass
