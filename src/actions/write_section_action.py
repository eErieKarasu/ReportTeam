from src.actions.base_action import BaseAction
from typing import Optional, List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class WriteSectionAction(BaseAction):
    """负责将分析结果转化为连贯文本的 Action"""

    @property
    def name(self) -> str:
        return "WriteSection"

    @property
    def description(self) -> str:
        return "根据提供的关键点和结构化数据，撰写报告章节的流畅文本。"

    async def run(self,
                  key_points: List[str],
                  structured_data: Optional[List[Dict[str, Any]]],
                  section_topic: str,
                  original_query: str,
                  language: str,
                  chart_filepath: Optional[str] = None) -> str:
        """
        执行章节撰写任务

        Args:
            key_points: 关键点列表
            structured_data: 结构化数据（可选）
            section_topic: 章节主题
            original_query: 原始查询
            language: 目标语言
            chart_filepath: 图表文件路径（可选）

        Returns:
            生成的章节文本
        """
        logger.info(
            f"WriteSectionAction: 开始为章节 '{section_topic}' 生成 {language} 文本 (基于 {len(key_points)} 个关键点)")

        # 获取语言名称
        lang_name = {'zh': '中文', 'en': 'English'}.get(language, 'English')

        # 将关键点合并为文本
        key_points_text = "\n - ".join(key_points) if key_points else ""

        # 准备结构化数据文本（如果有）
        structured_data_text = ""
        if structured_data and isinstance(structured_data, list) and len(structured_data) > 0:
            structured_data_text = "\n\n结构化数据:\n"
            for i, item in enumerate(structured_data):
                structured_data_text += f"{i + 1}. {str(item)}\n"

        # 构建写作提示
        from src.core import generate_text
        write_prompt = f"""Context: 用户查询是 '{original_query}'。你正在为报告撰写标题为 '{section_topic}' 的章节，使用 **{lang_name}** 语言。

任务: 基于以下关键点和结构化数据，撰写一个连贯、信息丰富、专业的章节文本。确保覆盖所有提供的要点，使它们自然地融合在一起。

风格指南:
- 使用清晰、专业的语言
- 保持段落结构和逻辑流畅
- 不要添加未在关键点中提供的信息
- 不要使用"根据提供的信息"或"根据关键点"等元引用
- 不要添加章节标题，直接开始正文内容

关键点:
 - {key_points_text}
{structured_data_text}

请生成 '{section_topic}' 章节的 {lang_name} 文本:
"""

        # 调用 LLM 生成文本
        logger.info(f"  正在调用 LLM 生成 '{section_topic}' 的 {lang_name} 文本...")
        generated_text = generate_text(write_prompt)

        if generated_text:
            logger.info(f"WriteSectionAction: 成功为章节 '{section_topic}' 生成文本 ({len(generated_text)} 字符)")
            return generated_text
        else:
            logger.error(f"WriteSectionAction: 无法生成章节 '{section_topic}' 的文本")
            return f"无法为章节 '{section_topic}' 生成文本。请稍后重试。"