import argparse
import importlib
import os
import sys
from pathlib import Path

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass, field
from typing import List, Optional

import yaml
from dotenv import load_dotenv

from agent.agent import Agent
from tools.tool_base import ToolRegistry, get_global_registry

try:
    import openai
except ImportError:
    openai = None

@dataclass
class Config:
    """A simple dataclass to hold the agent's configuration."""
    enabled_tools: List[str] = field(default_factory=list)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_turns: int = 10
    debug: int = 0
    sandbox: str = "local"
    sandbox_opts: dict = field(default_factory=dict)
    memory_backend: str = "faiss_store"
    memory_path: str = ".rag/index.faiss"
    embedding_model: str = "all-MiniLM-L6-v2"

def _load_config(cli_args: argparse.Namespace) -> Config:
    """Load config from YAML and merge CLI arguments."""
    with open("config/default.yaml") as f:
        yaml_config = yaml.safe_load(f)

    return Config(
        enabled_tools=yaml_config.get("enabled_tools", []),
        temperature=cli_args.temperature or yaml_config.get("temperature"),
        max_tokens=cli_args.max_tokens or yaml_config.get("max_tokens"),
        max_turns=cli_args.max_turns or yaml_config.get("max_turns"),
        debug=cli_args.debug,
        sandbox=yaml_config.get("sandbox", "local"),
        sandbox_opts=yaml_config.get("sandbox_opts", {}),
        memory_backend=yaml_config.get("memory_backend", "faiss_store"),
        memory_path=yaml_config.get("memory_path", ".rag/index.faiss"),
        embedding_model=yaml_config.get("embedding_model", "all-MiniLM-L6-v2"),
    )

def _load_tools(tool_names: List[str]) -> ToolRegistry:
    """Import tool modules and return the populated global registry."""
    for name in tool_names:
        importlib.import_module(f"tools.{name}")
    return get_global_registry()

def _setup_api_key() -> str:
    """Load OpenAI API key from .env.local or environment variables."""
    load_dotenv(dotenv_path=".env.local")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.", file=sys.stderr)
        print("Please set it in a .env.local file or as an environment variable.", file=sys.stderr)
        sys.exit(1)
    return api_key

def main():
    if openai is None:
        print("Error: The 'openai' package is not installed. Please run 'pip install openai'.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="The user prompt for the agent.")
    parser.add_argument("--debug", type=int, default=0, help="Set debug level.")
    parser.add_argument("--max-turns", type=int, default=None, help="Max Reason-Act turns.")
    parser.add_argument("--temperature", type=float, default=None, help="Override model temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max completion tokens.")
    args = parser.parse_args()

    # --- Initialization ---
    openai.api_key = _setup_api_key()
    config = _load_config(args)
    if hasattr(config, 'sandbox'):
        os.environ["SANDBOX_BACKEND"] = config.sandbox
    registry = _load_tools(config.enabled_tools)
    
    # --- Agent Execution ---
    agent = Agent(config, registry)
    agent.run(args.prompt)

if __name__ == "__main__":
    main()