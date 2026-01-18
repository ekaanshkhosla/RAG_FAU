# app/db/chroma_store.py

from chromadb import PersistentClient
from app.components.create_vector_store import create_embeddings

client = None
collection = None

def init_chroma(db_path: str, collection_name: str, create_embeddings_fn=None):

    """

    Initialize Chroma DB.

    - If collection exists and has data → reuse it

    - If empty → create embeddings

    """

    global client, collection
    client = PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    # Check if collection already has data
    count = collection.count()

    if count == 0:
        print("⚠️ Chroma collection is empty. Creating embeddings...")
        collection = create_embeddings()
    else:
        print(f"✅ Chroma collection loaded with {count} documents.")





def get_collection():
    if collection is None:
        raise RuntimeError("Chroma collection not initialized.")
    return collection 