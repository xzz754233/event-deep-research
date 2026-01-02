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


# FIXED: Helper function to run encoding in a thread
def _encode_text(text: str):
    enc = get_tokenizer()
    return enc.encode(text)


async def chunk_text_by_tokens(
    text: str, chunk_size: int = 1000, overlap_size: int = 20
) -> List[str]:
    """Splits text into token-based chunks asynchronously."""
    if not text:
        return []

    # FIXED: Run the CPU-bound encoding in a separate thread to avoid blocking the event loop
    tokens = await asyncio.to_thread(_encode_text, text)

    print(f"--- TOKENS: {len(tokens)} ---")

    chunks = []
    # Decode is fast enough to keep here, but encoding is the bottleneck
    encoding = get_tokenizer()

    start_index = 0
    while start_index < len(tokens):
        end_index = start_index + chunk_size
        chunk_tokens = tokens[start_index:end_index]
        chunks.append(encoding.decode(chunk_tokens))
        start_index += chunk_size - overlap_size

    print(f"--- Generated {len(chunks)} chunks ---")
    return chunks


async def count_tokens(messages: List[str]) -> int:
    """Counts tokens asynchronously."""
    combined = "".join(messages)
    tokens = await asyncio.to_thread(_encode_text, combined)
    return len(tokens)
