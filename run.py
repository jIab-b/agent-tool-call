import importlib
import os
import sys
from pathlib import Path

import sglang as sgl
import yaml

def _load_config(path: str = "config/default.yaml") -> dict:
    with open(Path(path)) as f:
        return yaml.safe_load(f)

def _load_tools(enabled):
    for name in enabled:
        importlib.import_module(f"tools.{name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py \"<your prompt>\"")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    cfg = _load_config()
    _load_tools(cfg.get("enabled_tools", []))

    model_path = os.path.expanduser(cfg["model_path"])
    tokenizer_path = os.path.expanduser(cfg["tokenizer_path"])

    # Verify the model path exists before trying to load it
    if not Path(model_path).is_dir():
        print(f"Error: Model path not found at '{model_path}'")
        print("Please ensure the path in config/default.yaml is correct and the model is downloaded.")
        sys.exit(1)

    runtime = sgl.Runtime(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    sgl.set_default_backend(runtime)

    from agent.agent import run_single_prompt
    run_single_prompt(prompt)

if __name__ == "__main__":
    main()