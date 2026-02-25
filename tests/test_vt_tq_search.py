import unittest
import sys
import os

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vt_tq_search.embedder import ToolEmbedder
from vt_tq_search.vector_store import ToolVectorStore
from vt_tq_search.query_engine import SemanticToolQuery

class MockFossilLogger:
    def __init__(self):
        self.events = []
    
    def log_tool_discovery(self, event):
        self.events.append(event)

class TestVTTQSearch(unittest.TestCase):
    def setUp(self):
        # Initialize components (will use mocks if deps missing)
        self.embedder = ToolEmbedder()
        self.vector_store = ToolVectorStore(force_mock=True)
        self.logger = MockFossilLogger()
        self.engine = SemanticToolQuery(self.embedder, self.vector_store, self.logger)
        
        # Test Data
        self.tools = [
            {
                "tool_id": "pinn_arbitration",
                "name": "PINN",
                "description": "Physics informed neural network",
                "capabilities": ["physics", "simulation"],
                "use_cases": ["simulate vehicle dynamics"]
            },
            {
                "tool_id": "docling_parser",
                "name": "Docling",
                "description": "Document parser",
                "capabilities": ["parse", "pdf"],
                "use_cases": ["read documents"]
            }
        ]
        
        # Index tools
        for tool in self.tools:
            emb = self.embedder.embed_tool(tool)
            self.vector_store.upsert_tool(tool["tool_id"], emb["embedding"], emb)

    def test_search_flow(self):
        """Test the end-to-end search flow."""
        query = "simulate vehicle physics"
        result = self.engine.find_tool(query)
        
        # Check structure
        self.assertIn("selected_tool", result)
        self.assertIn("confidence", result)
        self.assertIn("alternatives", result)
        self.assertIn("query_hash", result)
        
        print(f"\nQuery: {query}")
        print(f"Selected: {result['selected_tool']}")
        print(f"Confidence: {result['confidence']}")

    def test_logging(self):
        """Verify fossilization logging."""
        self.engine.find_tool("test query")
        self.assertEqual(len(self.logger.events), 1)
        event = self.logger.events[0]
        self.assertEqual(event["event_type"], "tool_discovery")
        self.assertEqual(event["query"], "test query")

if __name__ == '__main__':
    unittest.main()
