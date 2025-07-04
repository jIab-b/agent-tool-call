import unittest
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory.faiss_store import FaissStore

class TestFaissStore(unittest.TestCase):
    def setUp(self):
        self.index_path = ".test_rag/index.faiss"
        self.meta_path = ".test_rag/meta.pkl"
        self.store = FaissStore(index_path=self.index_path, meta_path=self.meta_path)

    def tearDown(self):
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        if os.path.exists(".test_rag"):
            os.rmdir(".test_rag")

    def test_ingest_and_query(self):
        async def _test():
            texts = ["this is a test", "this is another test"]
            metadata = [{"id": 1}, {"id": 2}]
            await self.store.ingest(texts, metadata)

            results = await self.store.query("a test")
            self.assertEqual(len(results), 2)
            self.assertEqual(len(results), 2)
            # Check if one of the returned metadata objects is {'id': 1}
            self.assertTrue(any(r['metadata'] == {'id': 1} for r in results))

        asyncio.run(_test())

if __name__ == "__main__":
    unittest.main()