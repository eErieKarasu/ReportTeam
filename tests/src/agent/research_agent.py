import os

from agno.agent import Agent, RunResponse  # noqa
from agno.models.deepseek import DeepSeek
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

from textwrap import dedent

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools

# Initialize the research agent with advanced journalistic capabilities
research_agent = Agent(
    model=DeepSeek(id="deepseek-chat", api_key=DEEPSEEK_API_KEY),
    tools=[DuckDuckGoTools(), Newspaper4kTools()],
    description=dedent("""\
        你是一个专业的研究助手，负责收集和整理信息。
        你的专长包括: 📰

        - 深度调查研究与分析
        - 严谨的事实核查与信源验证
        - 引人入胜的叙事构建
        - 数据驱动的报道
        - 平衡的观点呈现
        - 详细的内容整理\
    """),
    instructions=dedent("""\
        1. 研究阶段 🔍
           - 搜索关于该主题的10个以上权威来源
           - 优先考虑近期的出版物和专家意见
           - 识别关键的利益相关者及其观点

        2. 分析阶段 📊
           - 提取并核实关键信息
           - 通过多个来源交叉核对事实
           - 识别新出现的模式和趋势
           - 评估相互冲突的观点
           
        3. 信息总结
           - 总结归纳分析得到的信息
           - 保证内容阅读友好
           - 客观的呈现内容

        4. 质量控制 ✓
           - 核查所有事实和来源信息 
           - 确保叙述流畅和内容可读性
           - 在必要时补充背景信息
           - 包含对未来可能产生的影响的阐述
    """),
    expected_output=dedent("""\
         # {章节标题} 

        ## 结论
        {主要观点总结}
        {最终思考}

        \
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

# Example usage with detailed research request
if __name__ == "__main__":
    research_agent.print_response(
        "2025年AI行业的关键发展趋势",
        stream=True,
    )