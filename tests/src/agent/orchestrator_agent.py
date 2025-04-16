import os
from textwrap import dedent
from agno.agent import Agent
from dotenv import load_dotenv
from tests.src.agent.planner_agent import planner_agent
from tests.src.agent.research_agent import research_agent
from agno.models.deepseek import DeepSeek

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

agent_team = Agent(
    team=[planner_agent, research_agent],
    model=DeepSeek(id="deepseek-chat", api_key=DEEPSEEK_API_KEY),
    instructions=dedent("""\
        你是一家享有盛誉的新闻编辑部的主编! 📰

        你的职责:
        1. 协调planner_agent与research_agent的工作
        2. 将他们的发现整合成引人入胜的叙述
        3. 确保所有信息来源可靠且经过核实
        4. 呈现新闻与数据的平衡观点
        5. 突出关键风险与机遇

        你的风格指导:
        - 以引人注目的标题开头
        - 以强有力的内容提要（Executive Summary）开篇
        - 在不同类型信息之间使用清晰的章节分隔
        - 以“观察世界”和当前日期署名结尾\
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)

# Example usage with detailed research request
if __name__ == "__main__":
    research_agent.print_response(
        "生成一份关于“杀戮尖塔”的报告",
        stream=True,
    )