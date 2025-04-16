from .llm_integration import get_llm_client, BaseLLMClient
from .orchestration import Orchestrator
from typing import Optional

# 添加兼容性函数 generate_text，使用 get_llm_client
def generate_text(prompt: str) -> Optional[str]:
    """
    兼容性函数：使用新的 LLM 客户端接口执行文本生成。
    这个函数是为了兼容使用旧的 generate_text 函数的现有代码。
    
    Args:
        prompt: 发送给 LLM 的提示文本。
    
    Returns:
        LLM 生成的文本，如果生成失败则返回 None。
    """
    try:
        client = get_llm_client()
        return client.generate(prompt)
    except Exception as e:
        print(f"Error in compatibility generate_text function: {e}")
        return None

# Add other core components like StateManager, ErrorHandler later if needed
__all__ = ["get_llm_client", "Orchestrator", "generate_text", "BaseLLMClient"]