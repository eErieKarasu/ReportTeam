import logging
import asyncio
from typing import Any, Dict, List, Optional

from .base_action import BaseAction
from ..core.llm_integration import BaseLLMClient
# 导入新的网页获取工具
from ..tools.browser import fetch_multiple_pages_async

logger = logging.getLogger(__name__)


class BrowseAction(BaseAction):
    """
    用于浏览网页内容的 Action。
    使用 browser 工具获取指定 URL 的网页内容。
    """

    @property
    def name(self) -> str:
        return "browse"

    @property
    def description(self) -> str:
        return "获取指定 URL 的网页内容。输入应为完整的网页 URL 或 URL 列表。"

    def run(self, action_input: Any, llm_client: BaseLLMClient, **kwargs) -> Dict[str, Any]:
        """
        获取指定 URL 的网页内容。

        Args:
            action_input: 网页 URL（字符串）或 URL 列表。
            llm_client: LLM 客户端实例（本操作不直接使用）。
            **kwargs: 额外参数。

        Returns:
            包含获取到的网页内容的字典。
        """
        # 验证输入
        if isinstance(action_input, str):
            urls = [action_input.strip()]
        elif isinstance(action_input, list):
            urls = [url.strip() for url in action_input if isinstance(url, str) and url.strip()]
        else:
            return {
                "status": "Error",
                "error": "无效输入。预期为URL或URL列表。",
                "content": []
            }

        if not urls:
            return {
                "status": "Error",
                "error": "未提供有效URL。",
                "content": []
            }

        logger.info(f"执行'browse'操作，共{len(urls)}个URL。")

        try:
            successful_fetches = asyncio.run(fetch_multiple_pages_async(urls))

            if not successful_fetches:
                return {
                    "status": "Error",
                    "error": "无法从任何URL获取内容。",
                    "content": []
                }

            return {
                "status": "Success",
                "message": f"成功从{len(successful_fetches)}个URL获取内容。",
                "content": successful_fetches
            }

        except Exception as e:
            logger.error(f"执行'browse'操作时出错：{e}", exc_info=True)
            return {
                "status": "Error",
                "error": f"浏览操作异常：{e}",
                "content": []
            }