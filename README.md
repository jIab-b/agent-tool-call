# llm-tools-inference (prototype)

Minimal api-powered function-calling agent wired to a handful of demo tools.

```
python run.py
```

## Directory Layout
```
config/           # YAML configs (model path, enabled tools …)
tools/            # pluggable tools (auto-registered on import)
agent/            # conversation driver
run.py            # CLI entrypoint
bench.py          # original quick Qwen demo (unchanged)
download.py       # (kept as-is)
```

## Adding a New Tool

1. Create `tools/my_tool.py`:

```python
from tools.tool_base import Tool, register

def _run(args: dict) -> str:
    # do something
    return "result string"

register(Tool(
    name="my_tool",
    description="explain purpose",
    parameters={"type": "object", "properties": {}},
    run=_run,
))
```

2. Add `"my_tool"` to `enabled_tools` in `config/default.yaml` (or a runtime config).

The agent automatically exposes the tool; at inference time the model can respond with e.g.

```json
{"tool": "my_tool", "args": {...}}
```

and the dispatcher will execute `_run()` and feed the textual result back into the chat history.

## Extensibility Notes
* **LoRA / Adapters** – swap `model_path` & `tokenizer_path`, or augment runtime before `sgl.set_default_backend`.
* **Memory** – persist `conv` list from `agent/agent.py` to disk / vector DB.
* **Multimodality** – register an image tool, or run a multi-modal SGLang backend.
* **Remote MCP** – real implementation would invoke `use_mcp_tool` in `_run`.
