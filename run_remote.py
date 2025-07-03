import argparse
import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import sglang as sgl
import yaml

from agent.agent import Agent
from tools.tool_base import ToolRegistry, get_global_registry

# Optional OpenAI remote backend
try:
    import openai
except ImportError:
    openai = None

@dataclass
class Config:
    """A simple dataclass to hold the agent's configuration."""
    enabled_tools: List[str] = field(default_factory=list)
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_turns: int = 5
    debug: int = 0

def _load_config(cli_args: argparse.Namespace) -> Config:
    """Load config from YAML and merge CLI arguments."""
    with open("config/default.yaml") as f:
        yaml_config = yaml.safe_load(f)

    return Config(
        enabled_tools=yaml_config.get("enabled_tools", []),
        model_path=yaml_config.get("model_path"),
        tokenizer_path=yaml_config.get("tokenizer_path"),
        temperature=cli_args.temperature,
        max_tokens=cli_args.max_tokens,
        max_turns=cli_args.max_turns,
        debug=cli_args.debug,
    )

def _load_tools(tool_names: List[str]) -> ToolRegistry:
    """Import tool modules and return the populated global registry."""
    for name in tool_names:
        importlib.import_module(f"tools.{name}")
    return get_global_registry()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="The user prompt for the agent.")
    parser.add_argument("--debug", type=int, default=0, help="Set debug level.")
    parser.add_argument("--max-turns", type=int, default=5, help="Max Reason-Act turns.")
    parser.add_argument("--temperature", type=float, default=None, help="Override model temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max completion tokens.")
    args = parser.parse_args()

    config = _load_config(args)
    registry = _load_tools(config.enabled_tools)
    
    # --- Backend Setup ---
    runtime = None
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key and openai:
        openai.api_key = openai_api_key
    elif config.model_path:
        if not Path(config.model_path).is_dir():
            print(f"Error: Model path not found at '{config.model_path}'")
            sys.exit(1)
        runtime = sgl.Runtime(
            model_path=os.path.expanduser(config.model_path),
            tokenizer_path=os.path.expanduser(config.tokenizer_path),
        )
        sgl.set_default_backend(runtime)
    
    # --- Agent Execution ---
    agent = Agent(config, registry)
    agent.run(args.prompt)

    if runtime:
        runtime.shutdown()

if __name__ == "__main__":
    main()