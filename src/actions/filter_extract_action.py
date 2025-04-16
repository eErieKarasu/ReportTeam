from src.actions.base_action import BaseAction
from typing import Optional, List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


class FilterAndExtractRelevantAction(BaseAction):
    """筛选和提取相关信息的 Action"""

    @property
    def name(self) -> str:
        return "FilterAndExtractRelevant"

    @property
    def description(self) -> str:
        return "对原始的网页内容列表进行初步筛选，提取与当前主题直接相关的内容片段或初步关键点。"

    async def run(self, research_results: List[Dict[str, Any]], section_topic: str) -> List[str]:
        """
        对每个研究结果执行初步筛选和提取

        Args:
            research_results: 网页内容列表，每项包含 url 和 content
            section_topic: 当前主题

        Returns:
            提取出的初步关键点列表
        """
        logger.info(
            f"FilterAndExtractRelevant: 开始筛选与提取关于 '{section_topic}' 的相关要点 (处理 {len(research_results)} 个页面)")

        all_intermediate_points = []

        for i, page_data in enumerate(research_results):
            url = page_data.get('url', 'N/A')
            page_text = page_data.get('content', '')
            if not page_text:
                continue

            logger.debug(f"分析页面 {i + 1}/{len(research_results)}: {url} (文本长度: {len(page_text)})")

            # 对过长文本进行截断处理
            max_chars_per_page = 15000
            if len(page_text) > max_chars_per_page:
                logger.warning(f"截断页面文本，从 {len(page_text)} 字符到 {max_chars_per_page} 字符")
                page_text = page_text[:max_chars_per_page]

            # 构建提取关键点的提示
            from src.core import generate_text
            map_prompt = f"""Context: 用户询问关于 '{section_topic}' 的信息。
分析以下从 URL '{url}' 获取的文本内容。
提取与主题 '{section_topic}' 特别相关的关键点或见解。
以编号列表形式呈现。如果在此文本中没有找到相关点，请仅回复单词"None"。

文本内容:
---
{page_text}
---

与主题 '{section_topic}' 相关的关键点:
1."""

            # 调用 LLM 提取关键点
            logger.debug(f"  正在为页面 {i + 1} 提取关键点...")
            intermediate_analysis = generate_text(map_prompt)
            logger.debug(f"  页面 {i + 1} 提取的原始响应:\n<<<<<\n{intermediate_analysis}\n>>>>>")

            # 解析 LLM 响应
            if intermediate_analysis and intermediate_analysis.strip().lower() != "none":
                parsed_points = self._parse_llm_list_output(intermediate_analysis)
                if parsed_points:
                    logger.debug(f"  从页面 {i + 1} 提取了 {len(parsed_points)} 个关于 '{section_topic}' 的关键点")
                    all_intermediate_points.extend(parsed_points)
                else:
                    logger.warning(f"  无法从页面 {i + 1} 的 LLM 响应中解析列表点")
            else:
                logger.debug(f"  LLM 对页面 {i + 1} 关于主题 '{section_topic}' 的响应为 'None' 或空")

        logger.info(f"FilterAndExtractRelevant: 完成筛选提取，总共找到 {len(all_intermediate_points)} 个初步关键点")
        return all_intermediate_points

    def _parse_llm_list_output(self, text: Optional[str]) -> List[str]:
        """解析 LLM 输出的编号/项目符号列表"""
        if not text:
            return []
        points = [point.strip() for point in re.split(r'\n\s*(?:\d+\.|\*|-)\s+', text.strip()) if point.strip()]
        if points and re.match(r'^(?:\d+\.|\*|-)\s*', points[0]):
            points[0] = re.sub(r'^(?:\d+\.|\*|-)\s*', '', points[0]).strip()
        return [p for p in points if p]