import json
import os
from typing import List, Dict, Any

import sglang as sgl
from tools.tool_base import get, list_available
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

def _generate_reply(prompt: str) -> str:
    """
    Generate a reply using OpenAI 'o4-mini' when an API key is present;
    otherwise fall back to the local SGLang runtime.
    """
    if openai_api_key and openai is not None:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1024,
            temperature=1.0,
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
    debug_mode = kwargs.get("debug", False)
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
                    
                    if debug_mode:
                        print(f"--- Used Tool: {tool.name}, Args: {payload.get('args', {})} ---")

                    conv.append(("assistant", reply))
                    # FIX: Use the generic "tool" role instead of the specific tool name.
                    conv.append(("tool", result))
                else:
                    print("\nAssistant:")
                    print(reply)
                    break

            except (json.JSONDecodeError, KeyError, Exception) as e:
                if debug_mode:
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


def generate_plan(prompt_text: str, debug_mode: bool = False) -> List[Dict[str, Any]]:
    """Ask the model to return a `plan` array of tool calls."""
    system_prompt = build_system_prompt()
    conv = [("user", prompt_text)]
    reply = _generate_reply(make_history(conv, system_prompt))
    if debug_mode:
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


def run_single_prompt(prompt_text: str, **kwargs):
    """Hybrid planner-executor implementation."""
    debug_mode = kwargs.get("debug", False)
    try:
        plan = generate_plan(prompt_text, debug_mode=debug_mode)
    except Exception:
        # fallback to legacy behaviour if plan generation fails
        return _run_single_prompt_legacy(prompt_text, **kwargs)
    if not plan:
        return _run_single_prompt_legacy(prompt_text, **kwargs)

    conv = [("user", prompt_text)]
    outputs: List[Any] = []

    for idx, step in enumerate(plan, 1):
        tool_name = step.get("tool")
        if not tool_name:
            conv.append(("error", f"step {idx} missing tool"))
            break

        raw_args = step.get("args", {})
        args = _substitute_args(raw_args, outputs)

        try:
            tool = get(tool_name)
            result = tool.run(args)
        except Exception as e:
            conv.append(("error", f"{tool_name} failed: {e}"))
            if debug_mode:
                print(f"tool {tool_name} failed: {e}")
            break

        outputs.append(result)
        conv.append(("tool", result))
        if debug_mode:
            print(f"--- step {idx}: {tool_name}, args: {args} ---")

    # final answer
    final_reply = _generate_reply(make_history(conv, build_system_prompt()))
    print("\nAssistant:")
    print(final_reply)