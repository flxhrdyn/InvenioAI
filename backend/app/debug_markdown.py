import asyncio
import os
from app.retriever import build_retriever, retrieve_documents_async

async def debug_retrieval():
    print(f"--- Debugging Retrieval for 'Cash Flow Table' ---")
    
    # 1. Build the real retriever stack
    retriever, vectorstore, client = build_retriever()
    
    # 2. Search specifically for words that SHOULD be in the Page 20 table
    # Using the exact same query that failed before
    query = "Operating cash flow before change in net working capital 2023"
    
    docs, meta = await retrieve_documents_async(query, retriever, client)
    
    print(f"\nFound {len(docs)} documents.")
    
    for i, doc in enumerate(docs):
        page = doc.metadata.get('page_label', 'Unknown')
        print(f"\n[Result {i+1}] - Page: {page}")
        print("-" * 30)
        # Check if the content has table markers "|"
        content = doc.page_content
        is_table = "|" in content
        print(f"Table Detected: {is_table}")
        print(content[:800])
        print("-" * 30)

if __name__ == "__main__":
    # Ensure env is set for local testing
    os.environ["PYTHONPATH"] = "backend"
    asyncio.run(debug_retrieval())
