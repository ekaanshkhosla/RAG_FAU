from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"
BASE_DIR = Path(__file__).resolve().parents[2]

DB_NAME = BASE_DIR / "RAG_ChromaDB"
collection_name = "docs"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "transformed_data_llm_cleaned"
AVERAGE_CHUNK_SIZE = 500

RETRIEVAL_K = 10

openai = OpenAI()