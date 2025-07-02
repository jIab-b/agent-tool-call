import importlib
import argparse
import os
import sys
from pathlib import Path
 
import sglang as sgl
import yaml
 
# Optional OpenAI remote backend
try:
    import openai
except ImportError:
    openai = None

def _load_config(path: str = "config/default.yaml") -> dict:
    with open(Path(path)) as f:
        return yaml.safe_load(f)

def _load_tools(enabled):
    for name in enabled:
        importlib.import_module(f"tools.{name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="The user prompt for the agent.")
    parser.add_argument(
        "--debug", type=int, default=0, help="Set the debug level (0, 1, or 2)."
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum Reason-Act turns before giving up (default: 5).",
    )
    args = parser.parse_args()

    prompt = args.prompt
    debug_mode = args.debug
    max_turns = args.max_turns
    
    cfg = _load_config()
    _load_tools(cfg.get("enabled_tools", []))
 
    # --- Remote backend via OpenAI API if key provided ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        if openai is None:
            print("Please install the openai package (pip install openai) to use the remote model.")
            sys.exit(1)
        openai.api_key = openai_api_key
        from agent.agent import run_single_prompt
        run_single_prompt(prompt, debug=debug_mode, max_turns=max_turns)
        return

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
    run_single_prompt(prompt, debug=debug_mode, max_turns=max_turns)
    runtime.shutdown()

if __name__ == "__main__":
    main()