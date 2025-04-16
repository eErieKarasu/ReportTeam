import logging
from typing import Any, Dict, List, Optional

from .base_action import BaseAction
from ..core.llm_integration import BaseLLMClient

logger = logging.getLogger(__name__)

class SearchAction:
    """搜索操作，用于在互联网上查找信息。"""

    def __init__(self, max_results: int = 5):
        """
        初始化搜索操作。
        
        Args:
            max_results: 返回的最大结果数量。
        """
        self.name = "search"
        self.description = "搜索互联网上的信息"
        self.max_results = max_results

    def run(self, query, llm_client=None) -> Dict[str, Any]:
        """
        执行搜索操作。
        
        Args:
            query: 搜索查询。
            llm_client: LLM客户端（此操作不需要）。
            
        Returns:
            搜索结果字典。
        """
        # 移除查询中的引号，改善搜索结果
        query = query.replace('"', '').replace("'", '')
        
        logger.info(f"Executing 'search' action for query: '{query}' (max results: {self.max_results})")
        
        try:
            # 导入搜索工具
            from ..tools.web_search import simple_web_search
            
            # 执行搜索
            results = simple_web_search(query)
            
            if results and len(results) > 0:
                logger.info(f"Search for '{query}' returned {len(results)} results.")
                return {
                    "status": "Success",
                    "message": f"Found {len(results)} results for query: '{query}'",
                    "results": results
                }
            else:
                logger.warning(f"Search for '{query}' returned no results.")
                return {
                    "status": "Warning",
                    "message": "Search returned no results.",
                    "results": []
                }
        except Exception as e:
            logger.error(f"Error executing search for '{query}': {str(e)}", exc_info=True)
            return {
                "status": "Error",
                "message": f"Error executing search: {str(e)}",
                "results": []
            }