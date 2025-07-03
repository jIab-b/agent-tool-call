import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import sglang as sgl
from tools.tool_base import ToolRegistry, Tool

# Conditional imports for backend support
try:
    import openai
except ImportError:
    openai = None

class Agent:
    """An autonomous agent that uses LLMs and tools to solve tasks."""

    def __init__(self, config, registry: ToolRegistry):
        self.config = config
        self.registry = registry
        self.system_prompt = self._build_system_prompt()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def run(self, prompt_text: str):
        """Entry point to run the agent's reasoning loop."""
        return asyncio.run(self._run_reason_act_loop_async(prompt_text))

    def _build_system_prompt(self) -> str:
        """Dynamically builds the system prompt from the tool registry."""
        prompt = (
            "You are a helpful assistant. You have access to the following tools.\n"
            "When a function is needed, respond with a JSON *array* called `plan`, "
            "where each item is {\"tool\": \"<name>\", \"args\": {...}}.\n\n"
            "Here are the available tools:\n"
        )
        for tool in self.registry.list_available():
            prompt += f"- tool: {tool.name}\n"
            prompt += f"  description: {tool.description}\n"
            if "properties" in tool.parameters:
                props = tool.parameters["properties"]
                if props:
                    prompt += "  args:\n"
                    for name, info in props.items():
                        prompt += f"    {name}: {info.get('description', '')}\n"
        return prompt

    def _make_history(self, conv: List[tuple[str, str]]) -> str:
        """Formats the conversation history and prepends the system prompt."""
        out = ""
        for role, text in conv:
            out += f"{'tool_output' if role == 'tool' else role}: {text}\n"
        return self.system_prompt + "\n" + out

    def _generate_reply(self, prompt: str) -> str:
        """Generates a reply using the configured backend (OpenAI or local SGLang)."""
        if self.openai_api_key and openai:
            response = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens or 1024,
                temperature=self.config.temperature if self.config.temperature is not None else 1.0,
            )
            return response.choices[0].message.content.strip()
        else:
            # Fallback to local SGLang model
            state = sgl.run(sgl.user(prompt) + sgl.assistant(sgl.gen("reply", max_tokens=self.config.max_tokens or 1024)))
            return state["reply"].strip()

    async def _run_reason_act_loop_async(self, prompt_text: str):
        """The main async reasoning loop for the agent."""
        conv = [("user", prompt_text)]
        outputs = []
        for turn in range(1, self.config.max_turns + 1):
            if self.config.debug >= 2:
                print(f"--- Turn {turn}/{self.config.max_turns}: Generating plan ---")
            
            prompt = self._make_history(conv)
            reply = await asyncio.to_thread(self._generate_reply, prompt)
            
            if self.config.debug >= 2:
                print(f"--- LLM Raw Reply ---\n{reply}\n---------------------")

            plan = self._safe_extract_plan(reply)
            if plan is None:
                print(f"\nAssistant:\n{reply}")
                return

            conv.append(("assistant", reply))
            outputs = await self._execute_plan_async(plan, outputs)
            for result in outputs:
                conv.append(("tool", str(result)))

        print("\nAssistant:\nReached maximum reasoning turns without a definitive answer.")

    async def _execute_plan_async(self, plan: List[Dict[str, Any]], prev_outputs: List[Any]) -> List[Any]:
        """Executes a list of tool calls concurrently."""
        if self.config.debug >= 2:
            print(f"--- Executing plan with {len(plan)} steps ---")

        tasks = []
        for step in plan:
            tool_name = step.get("tool")
            args = self._substitute_args(step.get("args", {}), prev_outputs)
            if self.config.debug >= 1:
                print(f"--- Tool Call: {tool_name}({json.dumps(args)}) ---")
            tasks.append(asyncio.create_task(self.registry.invoke(self.registry.get(tool_name), args)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            if self.config.debug >= 1:
                print(f"--- Tool Result[{i}]: {res} ---")
        return results

    def _safe_extract_plan(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Safely extracts a JSON array plan from the LLM's reply."""
        try:
            start = text.find("[")
            if start == -1: return None
            arr, _ = json.JSONDecoder().raw_decode(text[start:])
            return arr if isinstance(arr, list) else None
        except (json.JSONDecodeError, ValueError):
            return None

    def _substitute_args(self, args: Dict[str, Any], outputs: List[Any]) -> Dict[str, Any]:
        """Replaces placeholders like '$1.output' with previous tool outputs."""
        def _sub(val):
            if isinstance(val, str) and val.startswith("$") and val.endswith(".output"):
                try:
                    idx = int(val[1:-7]) - 1
                    return outputs[idx]
                except (ValueError, IndexError):
                    return val
            return val
        return {k: _sub(v) for k, v in args.items()}