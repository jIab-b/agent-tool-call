import json
import re
import urllib.parse
import urllib.request

from .tool_base import Tool, register


def _fetch(query: str, num: int = 5) -> list[dict]:
    """
    Simple DuckDuckGo HTML scrape (no API key required).
    Returns a list of {title,url,snippet}.
    """
    url = "https://duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/117 Safari/537.36"
        )
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=5) as resp:
        html = resp.read().decode(errors="ignore")

    # crude parse for results
    results = []
    pattern = re.compile(r'<a rel="nofollow" class="result__a" href="([^"]+?)".*?>(.+?)</a>', re.S)
    for m in pattern.finditer(html):
        href, title = m.groups()
        title = re.sub(r"<.*?>", "", title)
        results.append({"title": title, "url": href})
        if len(results) >= num:
            break
    return results


def _run(args: dict) -> str:
    query = args["query"]
    top_k = int(args.get("k", 5))
    return json.dumps(_fetch(query, top_k), ensure_ascii=False)


register(
    Tool(
        name="web_search",
        description="search the web via DuckDuckGo (no API key)",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "how many results", "default": 5},
            },
            "required": ["query"],
        },
        run=_run,
    )
)