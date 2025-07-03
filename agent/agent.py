import json
import os
from typing import List, Dict, Any

import sglang as sgl
from tools.tool_base import get, list_available

# optional runtime JSON-schema validation
try:
    import jsonschema  # type: ignore
except ImportError:
    jsonschema = None

try:
    import openai
except ImportError:
    openai = None

def build_system_prompt():
    """
    Dynamically builds the system prompt by listing all available tools.
    """
    prompt = (
        "you are a helpful assistant. You have access to the following tools.\n"
        "When a function is needed, respond with a JSON *array* called `plan`, "
        "where each item is {\"tool\": \"<name>\", \"args\": {...}}.\n\n"
        "Here are the available tools:\n"
    )
    tools = list_available()
    for tool in tools:
        prompt += f"- tool: {tool.name}\n"
        prompt += f"  description: {tool.description}\n"
        if "properties" in tool.parameters:
            props = tool.parameters["properties"]
            if props:
                prompt += f"  args:\n"
                for param_name, param_info in props.items():
                    prompt += f"    {param_name}: {param_info.get('description', '')}\n"
    return prompt

@sgl.function
def chat(s, history):
    s += history + "assistant: " + sgl.gen("reply", max_tokens=1024)

# --- runtime selection between remote OpenAI and local SGLang backends ---
openai_api_key = os.getenv("OPENAI_API_KEY") if "OPENAI_API_KEY" in os.environ else None
if openai_api_key and openai is not None:
    openai.api_key = openai_api_key

def _generate_reply(
    prompt: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Generate a reply using OpenAI 'o4-mini' when an API key is present;
    otherwise fall back to the local SGLang runtime.
    """
    if openai_api_key and openai is not None:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens or 1024,
            temperature=temperature if temperature is not None else 1.0,
        )
        return response.choices[0].message.content.strip()
    else:
        state = chat.run(history=prompt)
        return state["reply"].strip()

def make_history(conv, system_prompt):
    """
    Formats the conversation history and prepends the system prompt.
    """
    out = ""
    for role, text in conv:
        if role == "tool":
            out += f"tool_output:\n{text}\n"
        else:
            out += f"{role}: {text}\n"
    return system_prompt + "\n" + out

def _run_single_prompt_legacy(prompt_text: str, **kwargs):
    """
    Runs the agent for a single turn with a given prompt.
    It will continue to loop through tools until a final text answer is generated.
    """
    debug_mode = kwargs.get("debug", 0)
    conv = [("user", prompt_text)]
    system_prompt = build_system_prompt()

    while True:
        prompt = make_history(conv, system_prompt)
        reply = _generate_reply(prompt)
        
        json_start_index = reply.find('{')
        
        if json_start_index != -1:
            try:
                json_string_slice = reply[json_start_index:]
                decoder = json.JSONDecoder()
                payload, _ = decoder.raw_decode(json_string_slice)
                
                if "tool" in payload and isinstance(payload, dict):
                    tool = get(payload["tool"])
                    result = tool.run(payload.get("args", {}))
                    
                    if debug_mode >= 1:
                        print(f"--- Used Tool: {tool.name}, Args: {payload.get('args', {})} ---")

                    conv.append(("assistant", reply))
                    # FIX: Use the generic "tool" role instead of the specific tool name.
                    conv.append(("tool", result))
                else:
                    print("\nAssistant:")
                    print(reply)
                    break

            except (json.JSONDecodeError, KeyError, Exception) as e:
                if debug_mode >= 2:
                    print(f"--- Tool Error: {e} ---")
                conv.append(("assistant", reply))
                conv.append(("error", f"Invalid tool call or execution error: {e}"))
        else:
            print("\nAssistant:")
            print(reply)
            break
# ---------------- Hybrid Planning & Execution Helpers ----------------

def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Extract the first JSON array found in `text`."""
    start = text.find("[")
    if start == -1:
        raise ValueError("no json array found")
    decoder = json.JSONDecoder()
    arr, _ = decoder.raw_decode(text[start:])
    if not isinstance(arr, list):
        raise ValueError("json is not an array")
    return arr


# -------------------------------------------------------------------------
# Safer plan extractor with optional schema validation
# -------------------------------------------------------------------------
def _safe_extract_plan(text: str) -> List[Dict[str, Any]] | None:
    """
    Attempt to extract a JSON array; return None on failure
    or schema violation (when jsonschema is available).
    """
    try:
        arr = _extract_json_array(text)
        if jsonschema is not None:
            jsonschema.validate(arr, {"type": "array"})
        return arr
    except Exception:
        return None


def generate_plan(
    prompt_text: str,
    *,
    debug_mode: int = 0,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    """Ask the model to return a `plan` array of tool calls."""
    system_prompt = build_system_prompt()
    conv = [("user", prompt_text)]
    reply = _generate_reply(
        make_history(conv, system_prompt),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if debug_mode >= 2:
        print("--- Plan LLM Reply ---")
        print(reply)
        print("----------------------")
    return _extract_json_array(reply)


def _substitute_args(args: Dict[str, Any], outputs: List[Any]) -> Dict[str, Any]:
    """Replace placeholders like '$1.output' with previous outputs."""
    def _sub(val):
        if isinstance(val, str) and val.startswith("$") and val.endswith(".output"):
            try:
                idx = int(val[1:-7]) - 1
                return outputs[idx]
            except Exception:
                return val
        return val
    return {k: _sub(v) for k, v in args.items()}


def is_plan_reply(text: str) -> bool:
    """Return True if `text` contains a JSON array plan."""
    try:
        _extract_json_array(text)
        return True
    except Exception:
        return False


def run_reason_act_loop(
    prompt_text: str,
    *,
    max_turns: int = 8,  # slightly higher headroom
    debug: int = 0,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Continuous Reason-Act loop.
    The agent repeatedly queries the LLM, executes any returned plan,
    and stops when a natural-language answer is provided or `max_turns`
    is exceeded.
    """
    system_prompt = build_system_prompt()
    conv: List[tuple[str, str]] = [("user", prompt_text)]

    for turn in range(1, max_turns + 1):
        if debug >= 2:
            print(f"--- Turn {turn}/{max_turns}: prompting LLM ---")
        reply = _generate_reply(
            make_history(conv, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if debug >= 2:
            print(f"--- LLM Reply ---\n{reply}\n-----------------")

        # Attempt to parse the reply as a plan.
        plan = _safe_extract_plan(reply)
        if plan is None:
            # Send corrective prompt and retry this turn
            if debug >= 1:
                print("--- Invalid or missing plan; requesting correction ---")
            conv.append(("assistant", reply))
            conv.append(
                (
                    "user",
                    "Your previous answer was not a valid JSON array named `plan`. "
                    "Please return ONLY the array following the specification.",
                )
            )
            continue
 
        # -----------------------------------------------------------------
        # The reply **is** a plan – execute each tool step.
        # -----------------------------------------------------------------
        conv.append(("assistant", reply))
        if debug >= 2:
            print(f"--- Detected plan with {len(plan)} steps ---")
 
        outputs: List[Any] = []
        for idx, step in enumerate(plan, 1):
            tool_name = step.get("tool")
            raw_args = step.get("args", {})
            args = _substitute_args(raw_args, outputs)
 
            if debug >= 1:
                print(f"--- Executing step {idx}: {tool_name}, args: {args} ---")
 
            try:
                tool = get(tool_name)
                result = tool.run(args)
            except Exception as e:
                result = f"Error executing {tool_name}: {e}"
                if debug >= 2:
                    print(f"--- Tool error: {result} ---")
 
            outputs.append(result)
            conv.append(("tool", result))
        # Loop continues to ask the LLM with updated history

    # Reached maximum turns without a final answer
    print("\nAssistant:")
    print("Reached maximum reasoning turns without a definitive answer.")


# -------------------------------------------------------------------------
# Back-compatibility shim
# -------------------------------------------------------------------------
def run_single_prompt(prompt_text: str, **kwargs):
    """
    Alias to run_reason_act_loop so existing entry points remain valid.
    Accepts 'debug', 'max_turns', 'temperature', and 'max_tokens' via **kwargs.
    """
    debug_mode = kwargs.get("debug", 0)
    max_turns = kwargs.get("max_turns", 8)
    temperature = kwargs.get("temperature")
    max_tokens = kwargs.get("max_tokens")
    return run_reason_act_loop(
        prompt_text,
        max_turns=max_turns,
        debug=debug_mode,
        temperature=temperature,
        max_tokens=max_tokens,
    )