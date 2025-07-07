import asyncio
import subprocess
import sys
import shlex

# Add project root to the Python path to allow imports from other directories
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.faiss_store import FaissStore
from memory.memory_manager import MemoryManager

def run_agent(prompt: str) -> str:
    """Executes the agent script and captures its full output."""
    cmd = f"python llm/run_remote.py {shlex.quote(prompt)}"
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

async def main():
    """
    A CLI controller for interacting with the LLM agent continuously,
    enhanced with a multi-layered memory system.
    """
    print("LLM Agent Controller (with Memory)")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 30)

    # 1. Initialize the memory system
    long_term_memory = FaissStore()
    memory_manager = MemoryManager(long_term_memory)
    memory_manager.load()

    try:
        while True:
            # 2. Get user input
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            # 3. Construct a context-rich prompt
            augmented_prompt = await memory_manager.construct_prompt(user_input)
            
            # 4. Run the agent with the augmented prompt
            print("Assistant is thinking...")
            agent_output = run_agent(augmented_prompt)
            print(f"Assistant: {agent_output}")

            # 5. Update memory with the new interaction
            await memory_manager.add_message("user", user_input)
            await memory_manager.add_message("assistant", agent_output)

            print("\n" + "-" * 30)

    except (KeyboardInterrupt, EOFError):
        print("\nExiting controller.")
    finally:
        # 6. Ensure memory is saved on exit
        print("Saving memory...")
        memory_manager.save()
        print("Done.")

if __name__ == "__main__":
    asyncio.run(main())