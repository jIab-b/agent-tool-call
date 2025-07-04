import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sandbox.local import LocalSandbox

class TestLocalSandbox(unittest.TestCase):
    def setUp(self):
        self.sandbox = LocalSandbox()

    def test_simple_execution(self):
        result = self.sandbox.run_code('print("hello")')
        self.assertEqual(result["stdout"], "hello\n")
        self.assertEqual(result["stderr"], "")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])

    def test_error(self):
        result = self.sandbox.run_code('raise ValueError("test error")')
        self.assertIn("ValueError: test error", result["stderr"])
        self.assertNotEqual(result["exit_code"], 0)

    def test_timeout(self):
        result = self.sandbox.run_code('import time; time.sleep(2)', timeout=1)
        self.assertTrue(result["timed_out"])
        self.assertIn("timed out", result["stderr"].lower())

if __name__ == "__main__":
    unittest.main()