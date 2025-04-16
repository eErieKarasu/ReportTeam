import logging
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from typing import Optional, List, Union

logger = logging.getLogger(__name__)

async def fetch_multiple_pages_async(urls: List[str]) -> List[dict]:
    """
    异步并行获取多个网页的内容
    
    Args:
        urls: 要获取内容的网页URL列表
        
    Returns:
        包含URL和内容的字典列表，仅包含成功获取的页面
    """
    logger.info(f"正在并行获取 {len(urls)} 个URL的内容")
    
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        stream=True
    )
    
    results = []
    
    try:
        async with AsyncWebCrawler() as crawler:
            run_conf = run_conf.clone(stream=False)
            crawl_results = await crawler.arun_many(urls, config=run_conf)

            for res in crawl_results:
                if res.success:
                    results.append({
                        "url": res.url,
                        "content": res.markdown.raw_markdown
                    })
                    logger.info(f"成功从 {res.url} 获取内容（长度：{len(res.markdown.raw_markdown)}）")
                else:
                    logger.warning(f"从 {res.url} 获取内容失败：{res.error_message}")

            # print(results)
            return results
            
    except Exception as e:
        logger.error(f"批量获取内容时发生异常：{e}", exc_info=True)
        return results

# 测试代码
async def _test_fetch():
    from src.utils import setup_logging
    setup_logging(log_level=logging.INFO)

    urls_to_test = [
        "https://docs.crawl4ai.com/core/quickstart/",
        "https://github.com/agno-agi/agno",
    ]
    
    # 测试多URL获取
    print(f"\n--- 测试 fetch_multiple_pages_async ---")
    results = await fetch_multiple_pages_async(urls_to_test)
    for res in results:
        print(f"成功从 {res['url']} 获取内容 (长度: {len(res['content'])})")

if __name__ == '__main__':
    asyncio.run(_test_fetch())
