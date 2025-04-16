import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class StateManager:
    """
    状态管理器，用于跟踪研究过程中的状态信息，
    提供结构化的状态记录和摘要，避免依赖LLM从原始历史中推断状态。
    """
    
    def __init__(self):
        self.current_section = None
        self.searches = []  # 已执行的搜索和结果
        self.browsed_urls = []  # 已浏览的URL
        self.browsed_contents = []  # 已浏览页面的内容摘要
        self.pending_urls = []  # 从搜索结果中提取但尚未浏览的URL
        self.final_answer = None  # 最终答案
        self.stage = "initial"  # 当前阶段: initial, searching, browsing, finalizing
    
    def set_current_section(self, section: str) -> None:
        """设置当前正在研究的章节。"""
        self.current_section = section
        logger.debug(f"StateManager: Current section set to '{section}'")
    
    def add_search_result(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        添加搜索结果到状态管理器。
        
        Args:
            query: 搜索查询
            results: 搜索结果列表
        """
        search_entry = {
            "query": query,
            "results": results,
            "timestamp": self._get_timestamp()
        }
        
        self.searches.append(search_entry)
        
        # 提取搜索结果中的URL添加到待浏览列表
        new_pending_urls = []
        for result in results:
            url = result.get("url")
            if url and url not in self.browsed_urls and url not in self.pending_urls:
                new_pending_urls.append({
                    "url": url,
                    "title": result.get("title", "无标题"),
                    "snippet": result.get("snippet", "")[:150]
                })
        
        self.pending_urls.extend(new_pending_urls)
        self.stage = "searching"
        
        logger.debug(f"StateManager: Added search results for query '{query}', found {len(results)} results")
        logger.debug(f"StateManager: Added {len(new_pending_urls)} new URLs to pending list")
    
    def add_browsed_content(self, urls: List[str], contents: List[Dict[str, Any]]) -> None:
        """
        添加浏览结果到状态管理器。
        现在基于实际获取到的内容来更新浏览状态和待处理列表。

        Args:
            urls: 期望浏览的URL列表 (来自LLM的action_input，主要用于日志记录或调试)
            contents: 成功获取的内容列表，每项包含成功获取的 'url' 和 'content'
        """
        successfully_browsed_urls = set() # 使用集合提高查找效率

        # 1. 处理成功获取的内容
        for content_data in contents:
            fetched_url = content_data.get("url")
            if not fetched_url:
                logger.warning("收到了没有URL的已浏览内容条目。")
                continue

            successfully_browsed_urls.add(fetched_url) # 记录成功浏览的URL

            # 如果尚未记录，则添加到 browsed_urls 列表
            if fetched_url not in self.browsed_urls:
                self.browsed_urls.append(fetched_url)

            # 添加内容摘要
            content_entry = {
                "url": fetched_url,
                # 取内容的开头部分作为摘要
                "content_summary": content_data.get("content", "")[:300] + ("..." if len(content_data.get("content", "")) > 300 else ""),
                "timestamp": self._get_timestamp()
            }
            # 避免在同一批次中为同一URL添加重复的内容条目（虽然不太可能）
            if not any(entry['url'] == fetched_url for entry in self.browsed_contents[-len(contents):]):
                 self.browsed_contents.append(content_entry)

        # 2. 从 pending_urls 中移除成功浏览的 URL
        original_pending_count = len(self.pending_urls)
        self.pending_urls = [
            item for item in self.pending_urls
            if item.get("url") not in successfully_browsed_urls
        ]
        removed_count = original_pending_count - len(self.pending_urls)

        # 如果有成功浏览的内容，更新阶段
        if successfully_browsed_urls:
            self.stage = "browsing"

        logger.debug(f"StateManager: 处理了 {len(successfully_browsed_urls)} 个URL的已浏览内容。")
        logger.debug(f"StateManager: 从待处理列表中移除了 {removed_count} 个URL。")
        logger.debug(f"StateManager: 剩余待处理URL数量: {len(self.pending_urls)}")
    
    def set_final_answer(self, answer: str) -> None:
        """设置最终答案。"""
        self.final_answer = answer
        self.stage = "finalizing"
        logger.debug("StateManager: Final answer set")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        获取当前状态的简洁摘要，用于注入到Prompt中。
        
        Returns:
            包含状态摘要的字典
        """
        summary = {
            "current_section": self.current_section,
            "stage": self.stage,
            "searches_performed": len(self.searches),
            "urls_browsed": len(self.browsed_urls),
            "pending_urls_count": len(self.pending_urls),
            "has_final_answer": self.final_answer is not None
        }
        
        # 添加最近的搜索查询和结果
        if self.searches:
            latest_search = self.searches[-1]
            summary["latest_search"] = {
                "query": latest_search["query"],
                "results_count": len(latest_search["results"]),
                "top_results": [
                    {
                        "title": result.get("title", "无标题"),
                        "url": result.get("url", ""),
                        "snippet": result.get("snippet", "")[:100] + "..." if len(result.get("snippet", "")) > 100 else result.get("snippet", "")
                    }
                    for result in latest_search["results"][:3]  # 只包含前3个结果
                ]
            }
        
        # 添加待浏览的URL
        if self.pending_urls:
            summary["pending_urls"] = [
                {
                    "url": item["url"],
                    "title": item["title"]
                }
                for item in self.pending_urls[:5]  # 只包含前5个待浏览URL
            ]
        
        # 添加已浏览的内容摘要
        if self.browsed_contents:
            summary["browsed_content_summaries"] = [
                {
                    "url": content["url"],
                    "summary": content["content_summary"][:150] + "..." if len(content["content_summary"]) > 150 else content["content_summary"]
                }
                for content in self.browsed_contents[-3:]  # 只包含最近浏览的3个内容
            ]
        
        return summary
    
    def should_search(self) -> bool:
        """判断是否应该执行搜索操作。"""
        # 如果已经执行了多次搜索但仍然没有结果，建议尝试更一般化的搜索词或直接进入finalizing
        if len(self.searches) >= 3 and len(self.browsed_urls) == 0:
            return False
        # 如果没有执行过搜索，或者有搜索但没有找到有用的URL，建议继续搜索
        return len(self.searches) == 0 or (len(self.searches) > 0 and len(self.pending_urls) == 0 and len(self.browsed_urls) == 0)
    
    def should_browse(self) -> bool:
        """判断是否应该执行浏览操作。"""
        # 如果有待浏览的URL，建议执行浏览操作
        return len(self.pending_urls) > 0
    
    def should_finalize(self) -> bool:
        """判断是否应该生成最终答案。"""
        # 如果已经浏览了足够的内容（至少1个页面），建议生成最终答案
        return len(self.browsed_contents) >= 1
    
    def get_recommended_action(self) -> Dict[str, Any]:
        """
        基于当前状态，提供下一步行动的建议。
        
        Returns:
            包含建议行动的字典
        """
        if self.should_search():
            return {
                "action": "search",
                "explanation": "尚未进行足够的搜索或之前的搜索未找到有用结果"
            }
        elif self.should_browse():
            urls_to_browse = [item["url"] for item in self.pending_urls[:3]]  # 建议一次最多浏览3个URL
            return {
                "action": "browse",
                "urls": urls_to_browse,
                "explanation": f"有{len(self.pending_urls)}个待浏览URL，建议浏览这些URL获取详细信息"
            }
        elif self.should_finalize():
            return {
                "action": "final_answer",
                "explanation": f"已浏览了{len(self.browsed_contents)}个页面的内容，可以整合信息生成最终答案"
            }
        else:
            return {
                "action": "search",
                "explanation": "状态不明确，建议执行新的搜索"
            }
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳，用于记录事件发生时间。"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def reset(self) -> None:
        """重置状态管理器，用于处理新的章节。"""
        self.current_section = None
        self.searches = []
        self.browsed_urls = []
        self.browsed_contents = []
        self.pending_urls = []
        self.final_answer = None
        self.stage = "initial"
        logger.debug("StateManager: State reset")