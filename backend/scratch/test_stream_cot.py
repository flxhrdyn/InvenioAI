import asyncio
import pytest
import json
from backend.app.rag_pipeline import rag_pipeline_stream_async

@pytest.mark.asyncio
async def test_stream():
    query = "Apa yang dimaksud dengan Universal Approximation Theorem?"
    chat_history = []
    
    print(f"Testing stream for query: {query}")
    async for chunk in rag_pipeline_stream_async(query, chat_history):
        data = json.loads(chunk)
        step = data.get("step")
        if step == "thinking":
            print(f"[THINKING] {data.get('content')}", end="", flush=True)
        elif step == "token":
            print(f"[TOKEN] {data.get('content')}", end="", flush=True)
        elif step == "done":
            print("\n[DONE]")
            # print(f"Full Answer: {data.get('answer')}")
            # print(f"Thoughts: {data.get('thoughts')}")
        else:
            print(f"\n[STEP] {step}")

if __name__ == "__main__":
    asyncio.run(test_stream())
