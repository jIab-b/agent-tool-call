from typing import Protocol, runtime_checkable

@runtime_checkable
class Sandbox(Protocol):
    def run_code(
        self,
        code: str,
        *,
        image: str | None = None,
        timeout: float = 5.0,
        memory: str | None = None,  # e.g. "512m"
        cpus: str | None = None,  # e.g. "0.5"
        network: bool = False,
    ) -> dict:
        """
        Executes code in a sandboxed environment.

        Returns:
            A dictionary containing stdout, stderr, exit_code, and timed_out status.
        """
        ...