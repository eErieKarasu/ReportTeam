import os
from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
from tests.src.agent.planner_agent import planner_agent
from tests.src.agent.research_agent import research_agent
from agno.models.deepseek import DeepSeek

from tests.src.agent.writer_agent import writer_agent

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

agent_team = Agent(
    team=[planner_agent, research_agent, writer_agent],
    # model=Gemini(id="gemini-2.0-flash-exp", api_key=GEMINI_API_KEY),
    model=DeepSeek(id="deepseek-chat", api_key=DEEPSEEK_API_KEY),
    instructions=dedent("""\
        - 角色：你是一家顶级媒体公司的资深主编。你的使命是指导团队产出一份基于充分研究、结构严谨、富有洞察力的高质量报告。
        你的手下有三个员工，分别是：
        - research_agent: 负责搜集资料
        - planner_agent: 负责生成框架大纲
        - writer_agent: 负责针对某一部分撰写文本
        你需要协调他们交付一份基于扎实研究、结构清晰、论证充分、可以直接发布的高质量报告文档。
        过程中的每一个环节你不满意，都可以重新协调分配任务。
        
        示例：
        query: 生成一份关于“人工智能”的报告。
        step1: 用户需要生成一份关于“人工智能”的报告，现在我没有相关的资料，我应该让research_agent去搜集“人工智能”的资料。
        step2: 现在我有了“人工智能”的资料，应该让planner_agent根据收集到的资料，给出一个报告的大纲。
        step3: 现在我有了大纲和资料，应该让writer_agent撰写文本。由于writer_agent一次只能写一个章节，我应该反复的调用直到报告完成。
        """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
    save_response_to_file='slay-the-spire.md'
)

# Example usage with detailed research request
if __name__ == "__main__":
    agent_team.print_response("生成一份关于“杀戮尖塔”的中文报告，报告包括三个章节，游戏简介、核心玩法、行业评价。", stream=True)