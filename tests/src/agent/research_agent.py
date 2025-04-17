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
        你的专长包括: 
        - 深度调查研究与分析
        - 严谨的事实核查与信源验证
        - 详细的内容整理\
    """),
    instructions=dedent("""\
        1. 研究阶段
           - 明确用户需求，发散性的思考并搜索相关内容
           - 资料要有足够的深度和广度
           
        2. 资料整理
           - 整理得到的信息，用段落的方式呈现
           - 保证内容阅读友好
           - 客观的呈现内容

        3. 质量控制
           - 核查所有事实和来源信息 
           - 请确保收集到的资料足够多，如果过少请重新搜索
    """),
    expected_output=dedent("""\
        {收集到的资料一}
        {收集到的资料二}
        {收集到的资料三}
        ···
        \
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

# Example usage with detailed research request
if __name__ == "__main__":
    research_agent.print_response(
        "杀戮尖塔",
        stream=True,
    )