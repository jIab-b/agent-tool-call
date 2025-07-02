# Hybrid Strategy Implementation Plan

This document outlines the changes required to introduce a selectable execution strategy (`reactive` vs. `hybrid`) into the agent.

## 1. Configuration (`config/default.yaml`)

We will add a new `strategy` key to the configuration file. This ensures backward compatibility by defaulting to the current reactive behavior if the key is absent.

**Proposed `config/default.yaml`:**
```yaml
# 'reactive' (default): The agent decides one step at a time.
# 'hybrid': The agent first creates a high-level plan, then executes it.
strategy: reactive

model_path: ~/huggingface/Qwen3-0.6B
tokenizer_path: ~/huggingface/Qwen3-0.6B
max_tokens: 1024
temperature: 0.7
enabled_tools:
  - file_search
  - file_read
  - web_search
  - mcp_wrapper
  - sandbox
  - list_directory
  - find_path
```

## 2. Agent Logic (`agent/agent.py`)

We will refactor `agent.py` to support multiple strategies without breaking existing functionality.

- **`run_single_prompt` will be renamed to `_run_reactive_strategy`**. Its code will remain unchanged.
- A new **`_run_hybrid_strategy`** function will be added to contain the planner/executor logic.
- A new public function, **`run`**, will act as a dispatcher, calling the appropriate strategy based on the configuration.

**Proposed `agent/agent.py` structure:**
```python
# ... (imports and helper functions remain)

def _run_reactive_strategy(prompt_text: str, **kwargs):
    # This function will contain the exact code from the original
    # run_single_prompt function.
    # ... existing while True loop ...

def _generate_plan(prompt_text: str) -> list[str]:
    # New function: Asks the LLM to create a high-level plan.
    # (Implementation to be added)
    print("--- Generating Plan ---")
    # Placeholder plan
    plan = [
        f"Understand the user's request: {prompt_text}",
        "Execute the necessary steps.",
        "Provide a final answer."
    ]
    return plan

def _execute_sub_goal(sub_goal: str, context: dict):
    # New function: The "Executor" loop to achieve one sub-goal.
    # (Implementation to be added, will be similar to the reactive loop)
    print(f"--- Executing Sub-Goal: {sub_goal} ---")
    # Placeholder result
    return f"Completed: {sub_goal}"


def _run_hybrid_strategy(prompt_text: str, **kwargs):
    # New function: The "Orchestrator" for the hybrid model.
    plan = _generate_plan(prompt_text)
    
    context = {} # Stores results from previous steps
    for sub_goal in plan:
        result = _execute_sub_goal(sub_goal, context)
        context[sub_goal] = result
    
    # Final answer generation would go here
    print("\nAssistant (Hybrid):")
    print("Plan executed. Final answer would be formulated here.")


def run(prompt: str, strategy: str = "reactive", **kwargs):
    """
    Main entrypoint for the agent.
    Dispatches to the correct strategy function based on the config.
    """
    if strategy == "hybrid":
        _run_hybrid_strategy(prompt, **kwargs)
    else:
        _run_reactive_strategy(prompt, **kwargs)

```

## 3. Entrypoint (`run_remote.py`)

The main script will be updated to read the `strategy` from the config and call the new `agent.run` dispatcher, instead of `run_single_prompt`.

**Proposed changes to `run_remote.py`'s `main` function:**
```python
# ... (imports and other functions)

def main():
    # ... (arg parsing)
    
    cfg = _load_config()
    _load_tools(cfg.get("enabled_tools", []))
    
    # Get the strategy from config, default to 'reactive'
    strategy = cfg.get("strategy", "reactive")

    # --- Backend setup (OpenAI or local SGLang) ---
    # ... (existing backend setup logic)

    # Import the new unified 'run' function
    from agent.agent import run
    
    # Call the dispatcher with the selected strategy
    run(prompt, strategy=strategy, debug=debug_mode)

    # ... (runtime shutdown if applicable)