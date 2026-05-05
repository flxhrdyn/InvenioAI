import time
import sys
import os

# Add current dir to path to import app
sys.path.append(os.getcwd() + "/backend")

from app.rag_pipeline import rag_pipeline
from app.config import CACHE_TYPE

print(f"Testing with CACHE_TYPE: {CACHE_TYPE}")

q = "What is Artificial Neural Network?"
h = []

print("\n--- First Call ---")
start = time.time()
r1 = rag_pipeline(q, h)
end = time.time()
print(f"Time: {end - start:.2f}s")

print("\n--- Second Call (Should be cached) ---")
start = time.time()
r2 = rag_pipeline(q, h)
end = time.time()
print(f"Time: {end - start:.2f}s")

if end - start < 0.1:
    print("\n✅ CACHE HIT SUCCESSFUL!")
else:
    print("\n❌ CACHE MISS! (Too slow for cache)")
