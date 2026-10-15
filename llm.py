"""llm.py - one place that decides which model provider is used.

If OLLAMA_URL is set in .env, everything (embeddings + chat) runs locally on Ollama, which speaks
the same API as OpenAI, so we only swap the base URL and model names. Otherwise OpenAI is used.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

USING_OLLAMA = bool(os.getenv("OLLAMA_URL"))

if USING_OLLAMA:
    client = OpenAI(base_url=os.environ["OLLAMA_URL"], api_key="ollama")  # key is required but ignored
    EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
else:
    client = OpenAI()  # reads OPENAI_API_KEY
    EMBED_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"

# Minimum cosine similarity (0-1) for a chunk to count as relevant. 0 = keep everything.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0"))
