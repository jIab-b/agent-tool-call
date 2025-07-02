import importlib
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
    debug_mode = "--debug" in sys.argv 
    args = [arg for arg in sys.argv if arg != '--debug']
    if len(sys.argv) < 2:
        print("Usage: python run.py \"<your prompt>\"")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
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
        run_single_prompt(prompt, debug=debug_mode)
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
    run_single_prompt(prompt, debug=debug_mode)
    runtime.shutdown()

if __name__ == "__main__":
    main()