from collections import deque
from typing import List, Dict, Any

from .base import Memory

class MemoryManager:
    """
    Manages a multi-layered memory system for an agent, combining a fixed-size
    short-term conversational buffer with a searchable long-term vector store.
    """
    def __init__(self, long_term_memory: Memory, max_history_size: int = 10):
        self.short_term_memory = deque(maxlen=max_history_size)
        self.long_term_memory = long_term_memory

    def load(self):
        """Loads the long-term memory from disk."""
        if hasattr(self.long_term_memory, 'load'):
            self.long_term_memory.load()

    def save(self):
        """Saves the long-term memory to disk."""
        if hasattr(self.long_term_memory, 'save'):
            self.long_term_memory.save()

    async def add_message(self, role: str, text: str):
        """
        Adds a message to the short-term history and ingests it into the
        long-term vector store.
        """
        self.short_term_memory.append({"role": role, "text": text})
        await self.long_term_memory.ingest([text], [{"role": role}])

    async def construct_prompt(self, query: str, k: int = 3) -> str:
        """
        Constructs an augmented prompt containing relevant context from both
        long-term and short-term memory.
        """
        # 1. Retrieve relevant documents from the long-term store
        retrieved_docs = await self.long_term_memory.query(query, k=k)
        
        # 2. Build the prompt string
        prompt = "### Context\n"
        prompt += "This is relevant information from past conversations:\n"
        for doc in retrieved_docs:
            prompt += f"- {doc.get('text', '')}\n"
        
        prompt += "\n### Conversation History\n"
        prompt += "This is the recent conversation history:\n"
        for message in self.short_term_memory:
            prompt += f"{message['role'].capitalize()}: {message['text']}\n"
            
        prompt += f"\n### User Query\n{query}"
        
        return prompt