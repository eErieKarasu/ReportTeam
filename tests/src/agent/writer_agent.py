import os

from agno.agent import Agent, RunResponse  # noqa
from agno.models.deepseek import DeepSeek
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

from textwrap import dedent
from agno.agent import Agent

# Initialize the research agent with advanced journalistic capabilities
writer_agent = Agent(
    model=DeepSeek(id="deepseek-chat", api_key=DEEPSEEK_API_KEY),
    instructions=dedent("""
    - 根据提供的信息，撰写对应的报告章节内容，不要随意的编造没有提到的信息。
    """),
    markdown=True,
    show_tool_calls=True,
)

# Example usage with detailed research request
if __name__ == "__main__":
    writer_agent.print_response(
        """
┃ ### 2.4 遗物系统的策略影响                                                                                                                                                   ┃
┃ 遗物分为**普通**、**稀有**、**Boss专属**三类，例如：                                                                                                                         ┃
┃ - **黑星**：击败精英怪后额外掉落遗物（高风险高回报）。                                                                                                                       ┃
┃ - **蛇眼**：每回合随机化手牌费用（适合特定流派）。   
        """,
        stream=True,
    )