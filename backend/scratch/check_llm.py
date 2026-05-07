import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def test_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found")
        return
    
    llm = ChatGroq(model="llama-3.1-70b-versatile", groq_api_key=api_key)
    try:
        res = llm.invoke("Hi")
        print(f"LLM Response: {res.content}")
    except Exception as e:
        print(f"LLM Error: {e}")

if __name__ == "__main__":
    test_llm()
