import json
import re
import urllib.parse
import urllib.request

from .tool_base import Tool, register


def _fetch(query: str, num: int = 5) -> list[dict]:
    """
    Uses the DuckDuckGo Instant Answer API (no API key required).
    Returns a list of {title, url}.
    """
    url = f"https://api.duckduckgo.com/?q={urllib.parse.quote_plus(query)}&format=json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        
        results = []
        if data.get("AbstractURL"):
            results.append({"title": data.get("Heading"), "url": data.get("AbstractURL")})

        for result in data.get("RelatedTopics", []):
            if "Text" in result and "FirstURL" in result:
                results.append({"title": result["Text"], "url": result["FirstURL"]})
            if len(results) >= num:
                break
        return results
    except Exception:
        return []


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