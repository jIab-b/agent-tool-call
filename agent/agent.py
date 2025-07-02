import json
import os
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
        "When a function is needed, respond with a json like {\"tool\": \"<name>\", \"args\": {...}}\n\n"
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
        # openai>=1.0.0 uses the `chat.completions.create` namespace
        response = openai.chat.completions.create(
            model="o4-mini",
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
        # FIX: Format tool output with a consistent 'tool' role for the model.
        if role == "tool":
            out += f"tool_output:\n{text}\n"
        else:
            out += f"{role}: {text}\n"
    return system_prompt + "\n" + out

def run_single_prompt(prompt_text: str, **kwargs):
    """
    Runs the agent for a single turn with a given prompt.
    It will continue to loop through tools until a final text answer is generated.
    """
    conv = [("user", prompt_text)]
    system_prompt = build_system_prompt()
    print(system_prompt)
    while True:
        prompt = make_history(conv, system_prompt)
        print(prompt)
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
                    
                    print(f"--- Used Tool: {tool.name}, Args: {payload.get('args', {})} ---")
                    print(result)
                    print("---")

                    conv.append(("assistant", reply))
                    # FIX: Use the generic "tool" role instead of the specific tool name.
                    conv.append(("tool", result))
                else:
                    print("\nAssistant:")
                    print(reply)
                    break

            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"--- Tool Error: {e} ---")
                conv.append(("assistant", reply))
                conv.append(("error", f"Invalid tool call or execution error: {e}"))
        else:
            print("\nAssistant:")
            print(reply)
            break