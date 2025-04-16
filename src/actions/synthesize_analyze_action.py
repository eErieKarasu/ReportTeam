from src.actions.base_action import BaseAction
from typing import Optional, List, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)


class SynthesizeAnalyzeAndAssessAction(BaseAction):
    """综合分析和评估的 Action"""

    @property
    def name(self) -> str:
        return "SynthesizeAnalyzeAndAssess"

    @property
    def description(self) -> str:
        return "对筛选后的关键点进行综合分析，提炼最终关键观点，提取结构化数据，并评估信息充分性。"

    async def run(self, relevant_information: List[str], section_topic: str, original_query: str, language: str) -> \
    Dict[str, Any]:
        """
        执行综合分析和评估

        Args:
            relevant_information: 初步提取的关键点列表
            section_topic: 当前主题
            original_query: 原始查询
            language: 目标语言

        Returns:
            包含 key_points, structured_data, sufficiency, suggestion_for_improvement 的字典
        """
        logger.info(
            f"SynthesizeAnalyzeAndAssess: 开始对 {len(relevant_information)} 个关键点进行综合分析 (主题: '{section_topic}', 语言: '{language}')")

        if not relevant_information:
            logger.warning(f"没有找到关于主题 '{section_topic}' 的关键点，无法进行综合分析")
            return {
                "sufficient": False,
                "key_points": [],
                "structured_data": None,
                "suggestion_for_further_research": f"需要重新搜索关于 '{section_topic}' 的基本信息"
            }

        # 将关键点合并为文本
        intermediate_points_text = "\n - ".join(relevant_information)

        # 截断过长的文本
        max_chars_for_reduce = 20000
        if len(intermediate_points_text) > max_chars_for_reduce:
            logger.warning(f"截断中间点文本，从 {len(intermediate_points_text)} 字符到 {max_chars_for_reduce} 字符")
            intermediate_points_text = intermediate_points_text[:max_chars_for_reduce]

        # 获取语言名称
        lang_name = {'zh': '中文', 'en': 'English'}.get(language, 'English')

        # 构建综合分析的提示
        from src.core import generate_text
        reduce_prompt = f"""Context: 用户查询是 '{original_query}'。正在为主题 '{section_topic}' 生成 {lang_name} 内容。
基于以下关于 '{section_topic}' 的提取要点进行分析。

任务:
1.  **综合和筛选:** 将要点整合为最终的、简洁的独特关键见解列表，用 **{lang_name}** 表述。
2.  **提取结构化数据:** 识别与 '{section_topic}' 相关的任何定量数据、比较或表格信息，可用于生成简单图表（如柱状图）。格式化为字典列表（例如：`[{{"类别": "A", "数值": 10}}, {{"类别": "B", "数值": 20}}]`）。如果找不到合适的结构化数据，使用 `null` 或空列表 `[]`。
3.  **评估充分性:** 批判性地评估综合的关键点（任务1）是否提供足够信息来撰写关于 '{section_topic}' 的全面章节。
4.  **建议进一步研究（如需要）:** 如果信息 *不充分*（任务3），用 {lang_name} 建议主题 '{section_topic}' 需要的具体信息。

请严格按 JSON 格式提供输出:
{{
  "sufficient": boolean,
  "key_points": [ // 用 {lang_name} 综合的要点列表
    "要点 1...",
    "要点 2..."
  ],
  "structured_data": [ // 用于图表的字典列表，无则为 null/[]
    {{"列名1": "类别A", "列名2": 数值1}},
    ...
  ],
  "suggestion_for_further_research": "建议文本..." // 仅当 sufficient 为 false 时
}}

示例（充分，有数据）:
```json
{{
  "sufficient": true,
  "key_points": ["要点 A ({lang_name})", "要点 B ({lang_name})"],
  "structured_data": [
    {{"产品": "手机", "市场份额": 40}},
    {{"产品": "电脑", "市场份额": 30}}
  ]
}}
```
示例（不充分，无数据）:
```json
{{
  "sufficient": false,
  "key_points": ["要点 C ({lang_name})"],
  "structured_data": [],
  "suggestion_for_further_research": "需要关于 {{具体方面}} 的更多数据 ({lang_name})。"
}}
```

关于主题 '{section_topic}' 的提取要点:
---
 - {intermediate_points_text}
---

现在，请为主题 '{section_topic}' 提供 JSON 输出:
"""

        # 调用 LLM 进行综合分析
        logger.info(f"  正在为主题 '{section_topic}' 进行最终 {lang_name} 综合、数据提取和充分性评估...")
        llm_assessment_response = generate_text(reduce_prompt)

        # 解析 LLM 响应
        analysis_assessment = self._parse_sufficiency_json_output(llm_assessment_response)

        if analysis_assessment:
            logger.info(
                f"SynthesizeAnalyzeAndAssess: 完成对主题 '{section_topic}' 的分析。充分: {analysis_assessment['sufficient']}, 找到数据: {analysis_assessment['structured_data'] is not None and len(analysis_assessment['structured_data']) > 0}")
            return analysis_assessment
        else:
            logger.error(f"SynthesizeAnalyzeAndAssess: 无法解析 LLM 对主题 '{section_topic}' 的分析/数据 JSON")
            return {
                "sufficient": False,
                "key_points": [],
                "structured_data": None,
                "suggestion_for_further_research": f"LLM 响应解析失败，需要重试分析 '{section_topic}'"
            }

    def _parse_sufficiency_json_output(self, text: Optional[str]) -> Optional[Dict[str, Any]]:
        """解析 LLM 的 JSON 输出"""
        if not text:
            logger.error("LLM 返回空响应")
            return None
        try:
            logger.debug(f"尝试解析 JSON: {text[:500]}...")
            if text.strip().startswith("```json"):
                text = text.strip()[7:-3].strip()
            elif text.strip().startswith("```"):
                text = text.strip()[3:-3].strip()

            parsed_json = json.loads(text)

            if not isinstance(parsed_json, dict):
                logger.error(f"解析的 JSON 不是字典: {parsed_json}")
                return None

            # 验证字段
            sufficient = parsed_json.get('sufficient')
            key_points = parsed_json.get('key_points')
            suggestion = parsed_json.get('suggestion_for_further_research')

            if not isinstance(sufficient, bool):
                return None
            if not isinstance(key_points, list) or not all(isinstance(item, str) for item in key_points):
                return None
            if suggestion is not None and not isinstance(suggestion, str):
                suggestion = None
            if sufficient and suggestion:
                suggestion = None
            if not sufficient and not suggestion:
                logger.warning("信息不充分但未提供建议")

            # 验证结构化数据
            structured_data = parsed_json.get('structured_data')
            if structured_data is not None:
                if not isinstance(structured_data, list):
                    logger.warning(f"'structured_data' 不是列表，忽略。值: {structured_data}")
                    structured_data = None
                elif not all(isinstance(item, dict) for item in structured_data):
                    logger.warning(f"'structured_data' 中不是所有项都是字典，忽略。值: {structured_data}")
                    structured_data = None

            validated_data = {
                'sufficient': sufficient,
                'key_points': key_points,
                'structured_data': structured_data,
                'suggestion_for_further_research': suggestion
            }

            points_count = len(key_points)
            data_status = f"{len(structured_data)} 项" if structured_data is not None else "无"
            logger.debug(
                f"成功解析分析 JSON: 充分={sufficient}, 要点={points_count}, 结构化数据={data_status}, 建议='{suggestion}'")
            return validated_data

        except json.JSONDecodeError as e:
            logger.error(f"无法解析 LLM 响应为 JSON: {e}. 响应内容:\n{text}")
            return None
        except Exception as e:
            logger.error(f"解析 LLM 响应时发生意外错误: {e}", exc_info=True)
            return None