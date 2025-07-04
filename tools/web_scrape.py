import requests
from bs4 import BeautifulSoup
from .tool_base import Tool, register

def _run(args: dict) -> str:
    """
    Fetches the content of a web page and returns the text.
    """
    url = args["url"]
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        return " ".join(soup.stripped_strings)
    except requests.exceptions.RequestException as e:
        return f'{{"error": "Failed to fetch URL: {url}", "details": "{e}"}}'

register(
    Tool(
        name="web_scrape",
        description="Fetches the text content of a given web page URL. Does not work for local file paths.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the web page to scrape.",
                },
            },
            "required": ["url"],
        },
        run=_run,
    )
)