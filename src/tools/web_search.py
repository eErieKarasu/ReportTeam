from gc import set_debug

import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin
import logging


def simple_web_search(query: str, num_results: int = 5) -> Optional[List[Dict[str, str]]]:
    """
    Performs a simple web search using DuckDuckGo (HTML version) and extracts results
    including title, snippet, and URL.

    Args:
        query: The search query string.
        num_results: The desired number of results.

    Returns:
        A list of dictionaries, where each dictionary contains 'title', 'snippet', and 'url',
        or None if the search failed or no results were found.
    """
    base_url = "https://html.duckduckgo.com/"
    search_url = f"{base_url}html/?q={requests.utils.quote(query)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    logging.info(f"Searching web for: {query}")

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # 尝试多种可能的结果块类名
        result_blocks = soup.find_all('div', class_='result__body', limit=num_results + 2)
        
        # 如果没有找到结果，尝试其他类名
        if not result_blocks:
            # 尝试新的类名或其他选择器
            result_blocks = soup.find_all('div', class_=['result', 'web-result'], limit=num_results + 2)
        
        # 如果仍然没有找到结果，尝试更通用的选择器
        if not result_blocks:
            # 查找包含链接和文本的div标签
            result_blocks = soup.find_all('div', limit=num_results * 3)

        if not result_blocks:
            logging.error("No result blocks found in HTML.")
            logging.debug(f"Page content: {soup.prettify()[:500]}...")  # 记录页面内容的一部分用于调试
            return None

        count = 0
        for block in result_blocks:
            # 尝试多种可能的类名
            title_tag = block.find('a', class_='result__a') or block.find('a', class_=['result-title', 'title'])
            snippet_tag = block.find('a', class_='result__snippet') or block.find('div', class_=['result-snippet', 'snippet'])
            
            # 如果没有找到特定类名的标签，尝试获取任何链接和文本
            if not title_tag:
                # 尝试找到第一个链接作为标题
                title_tag = block.find('a')
            
            if not snippet_tag:
                # 尝试找到一个包含文本的段落或div作为摘要
                snippet_tag = block.find(['p', 'div'], class_=None)

            if title_tag and title_tag.has_attr('href') and count < num_results:
                title = title_tag.get_text(strip=True)
                
                # 获取snippet文本，如果snippet_tag存在
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No snippet available"
                
                raw_url = title_tag['href']

                # 处理URL重定向
                if raw_url.startswith('/l/'):
                    try:
                        from urllib.parse import parse_qs, urlparse
                        parsed_link = urlparse(raw_url)
                        query_params = parse_qs(parsed_link.query)
                        if 'uddg' in query_params and query_params['uddg']:
                             url = query_params['uddg'][0]
                        else:
                             url = urljoin(base_url, raw_url)
                    except Exception:
                         url = urljoin(base_url, raw_url)
                else:
                    url = urljoin(base_url, raw_url)

                if title and url:
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url
                    })
                    count += 1

        if not results:
             logging.error("Could not extract valid results from result blocks.")
             logging.debug(f"Found {len(result_blocks)} blocks but couldn't extract results.")
             return None

        logging.info(f"Found {len(results)} potential results.")
        return results

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during web search request: {e}")
        return None
    except Exception as e:
        logging.error(f"Error parsing search results: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


# Example usage
if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    test_query = "人工智能 发展历程 关键里程碑 历史阶段 权威来源"
    search_results = simple_web_search(test_query)
    if search_results:
        logging.info("\n--- Search Results (with URLs) ---")
        for i, res in enumerate(search_results):
            logging.info(f"{i+1}. Title: {res['title']}")
            logging.info(f"   Snippet: {res['snippet']}")
            logging.info(f"   URL: {res['url']}\n")
    else:
        logging.error(f"Could not retrieve results for '{test_query}'")

    print(search_results)


