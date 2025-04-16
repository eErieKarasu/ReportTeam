# AI 自动化报告生成器 (Personal Learning Project)

这是一个我为了学习和探索 AI Agent、网络爬虫和自动化流程而构建的个人项目。

## 项目目标

本项目旨在实现一个自动化报告生成流程。用户输入一个查询（例如，“比较 2024 年第一季度手机品牌的市场份额”），系统将自动：

1.  **规划报告结构**：基于查询生成报告的大纲。
2.  **在线研究**：从网络上搜索和抓取相关信息。
3.  **数据分析**：处理和分析收集到的数据。
4.  **图表生成**：根据分析结果生成可视化图表。
5.  **报告撰写**：将所有信息整合成一份结构化的报告。
6.  **输出报告**：将最终报告保存为 Markdown 文件。

## 使用的技术栈

*   **核心流程编排**: Python, `asyncio`
*   **AI Agent**:
    *   **Planner, Analyzer, Writer**: 利用 `google-generativeai` (Gemini) 进行自然语言处理和内容生成。
    *   **Researcher**: 使用 `playwright` 和 `crawl4ai` 进行网页抓取和数据提取。
    *   **Chart Generator**: 使用 `matplotlib` 生成图表。
*   **数据处理**: `pandas`
*   **配置管理**: `python-dotenv`
*   **日志**: `logging`

## 我学到了什么

通过这个项目，我实践和学习了：

*   **AI Agent 的设计与协作**: 如何将一个复杂任务拆解给不同的 Agent，并通过 Orchestrator 进行协同工作。
*   **大型语言模型 (LLM) 的应用**: 如何使用 Gemini API 进行规划、分析和文本生成。
*   **异步编程**: 使用 `asyncio` 提高 I/O 密集型任务（如网络请求）的效率。
*   **Web 爬虫技术**: 学习了如何使用 `playwright` 等工具与动态网页交互并提取数据。
*   **数据处理与可视化**: 使用 `pandas` 和 `matplotlib` 对数据进行基础的处理和可视化展示。
*   **项目结构**: 如何组织一个包含多个模块（agents, tools, core, schemas）的 Python 项目。
*   **自动化流程**: 将多个步骤串联起来，实现从输入查询到输出报告的端到端自动化。

## 如何运行 (示例)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置环境变量 (例如 API 密钥)
# 创建 .env 文件并填入必要的 API 密钥，如 GOOGLE_API_KEY

# 3. 运行主程序并传入查询
python src/main.py "你的研究主题或查询"

# 示例:
# python src/main.py "2024 年电动汽车市场的主要趋势是什么？"
```

报告将生成在 `output/reports/` 目录下。

## 未来可能的改进

*   增强 Researcher Agent 的网页抓取能力和信息筛选能力。
*   引入更复杂的数据分析和可视化技术。
*   提供 Web UI 界面进行交互。
*   支持更多类型的输出格式。

---

**注意**: 这主要是一个学习项目，代码可能不完美，报告的准确性和完整性依赖于网络信息和 AI 模型的判断。