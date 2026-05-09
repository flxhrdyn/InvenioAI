import asyncio
import sys
import os
from pathlib import Path
import logging

# Add backend to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

from backend.app.rag_pipeline import rag_pipeline, rag_pipeline_stream_async
import json
import time

def test_sync_cache():
    print("\n--- Testing Sync Cache ---")
    question = "Apa itu ML?"
    history = []
    
    print(f"First call for '{question}'...")
    start = time.monotonic()
    res1 = rag_pipeline(question, history)
    duration1 = time.monotonic() - start
    print(f"First call took {duration1:.2f}s")
    
    # Simulate turn 2
    history = [(question, res1.get('answer'))]
    
    print(f"\nSecond call for '{question}' (with history)...")
    start = time.monotonic()
    res2 = rag_pipeline(question, history)
    duration2 = time.monotonic() - start
    print(f"Second call took {duration2:.2f}s")
    
    print(f"First answer: {res1.get('answer')[:100]}...")
    print(f"Second answer: {res2.get('answer')[:100]}...")
    
    if duration2 < 1.0:
        print("SUCCESS: Second call was served from cache (fast).")
    else:
        print("FAILURE: Second call was NOT served from cache (too slow).")

async def test_async_cache():
    print("\n--- Testing Async Cache ---")
    query = "Apa itu ML?"
    chat_history = []
    
    print(f"First call for '{query}'...")
    start = time.monotonic()
    answer1 = ""
    async for chunk in rag_pipeline_stream_async(query, chat_history):
        data = json.loads(chunk)
        if data.get("step") == "done":
            answer1 = data.get("answer")
    duration1 = time.monotonic() - start
    print(f"First call took {duration1:.2f}s")
    
    # Simulate turn 2
    chat_history = [f"user: {query}", f"assistant: {answer1}"]
    
    print(f"\nSecond call for '{query}' (with history)...")
    start = time.monotonic()
    cached_hit = False
    async for chunk in rag_pipeline_stream_async(query, chat_history):
        data = json.loads(chunk)
        if data.get("step") == "cached":
            cached_hit = True
    duration2 = time.monotonic() - start
    print(f"Second call took {duration2:.2f}s")
    
    if cached_hit:
        print("SUCCESS: Second call detected 'cached' step.")
    else:
        print("FAILURE: Second call did NOT detect 'cached' step.")

if __name__ == "__main__":
    test_sync_cache()
    asyncio.run(test_async_cache())
