import subprocess
import sys
import shlex

def main():
    """
    A CLI controller for interacting with the LLM agent continuously.
    This tool wraps the existing run_remote.py script to provide a chat-like interface.
    """
    print("LLM Agent Controller")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 30)

    while True:
        try:
            # 1. Get user input from the prompt
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting controller.")
                break

            if not user_input.strip():
                continue

            # 2. Construct the command to run the agent script
            # We use shlex.quote to ensure the user input is passed safely to the shell.
            cmd = f"python llm/run_remote.py {shlex.quote(user_input)}"

            # 3. Execute the command and stream the output in real-time
            # This allows us to see the agent's progress as it happens.
            with subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            ) as process:
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        print(line, end='')

            print("\n" + "-" * 30)

        except KeyboardInterrupt:
            print("\nExiting controller.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()