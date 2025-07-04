import unittest
import subprocess
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sandbox.docker import DockerSandbox

def is_docker_running():
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=2)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

@unittest.skipIf(not is_docker_running(), "Docker is not running or not installed.")
class TestDockerSandbox(unittest.TestCase):
    def setUp(self):
        self.sandbox = DockerSandbox()

    def test_simple_execution(self):
        result = self.sandbox.run_code('print("hello docker")')
        self.assertEqual(result["stdout"], "hello docker\n")
        self.assertEqual(result["stderr"], "")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])

    def test_error(self):
        result = self.sandbox.run_code('import sys; sys.exit(42)')
        self.assertEqual(result["exit_code"], 42)

    def test_timeout(self):
        result = self.sandbox.run_code('import time; time.sleep(2)', timeout=1)
        self.assertTrue(result["timed_out"])
        self.assertIn("timed out", result["stderr"].lower())
        
    def test_network_disabled(self):
        code = 'import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.connect(("google.com", 80))'
        result = self.sandbox.run_code(code, network=False)
        self.assertIn("Temporary failure in name resolution", result["stderr"])
        self.assertNotEqual(result["exit_code"], 0)

if __name__ == "__main__":
    unittest.main()