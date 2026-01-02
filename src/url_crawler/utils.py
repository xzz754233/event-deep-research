import asyncio
import os
import re
from typing import List
import aiohttp
import tiktoken

FIRECRAWL_API_URL = (
    f"{os.getenv('FIRECRAWL_BASE_URL', 'https://api.firecrawl.dev')}/v0/scrape"
)


async def url_crawl(url: str) -> str:
    """Crawls a URL and returns its content."""
    content = await scrape_page_content(url)
    if content is None:
        return ""

    # --- 唯一保留的必要優化：物理截斷 ---
    # 不要讓 40 萬字的文章進入後面的處理流程，直接在源頭砍斷。
    # 20,000 字元約等於 4000-5000 tokens，這對一篇新聞報導來說綽綽有餘。
    if len(content) > 20000:
        print(f"⚠️ Content too long ({len(content)} chars). Truncating to 20k.")
        content = content[:20000]

    return remove_markdown_links(content)


async def scrape_page_content(url):
    """Scrapes URL using Firecrawl API."""
    try:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                FIRECRAWL_API_URL,
                json={
                    "url": url,
                    "pageOptions": {"onlyMainContent": True},
                    "formats": ["markdown"],
                },
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("data", {}).get("markdown")
    except Exception as e:
        print(f"Error scraping page content: {e}")
        return None


def remove_markdown_links(markdown_text):
    return re.sub(r"\[(.*?)\]\(.*?\)", r"\1", markdown_text)


# Global tokenizer cache
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


# FIXED: 完全移除多線程 (asyncio.to_thread)，回到最原本的同步寫法
async def chunk_text_by_tokens(
    text: str, chunk_size: int = 1000, overlap_size: int = 20
) -> List[str]:
    """Splits text into token-based chunks synchronously."""
    if not text:
        return []

    # 直接計算，雖然會卡住 Main Loop 0.01秒，但絕對不會報錯
    encoding = get_tokenizer()
    tokens = encoding.encode(text)

    print(f"--- TOKENS: {len(tokens)} ---")

    chunks = []
    start_index = 0
    while start_index < len(tokens):
        end_index = start_index + chunk_size
        chunk_tokens = tokens[start_index:end_index]
        chunks.append(encoding.decode(chunk_tokens))
        start_index += chunk_size - overlap_size

    print(f"--- Generated {len(chunks)} chunks ---")
    return chunks


async def count_tokens(messages: List[str]) -> int:
    encoding = get_tokenizer()
    combined = "".join(messages)
    return len(encoding.encode(combined))
