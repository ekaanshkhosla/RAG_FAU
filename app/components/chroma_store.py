# app/db/chroma_store.py

from chromadb import PersistentClient
from app.components.create_vector_store import create_embeddings
from app.utils.logger import get_logger
from app.utils.custom_exception import CustomException

logger = get_logger(__name__)

client = None
collection = None

def init_chroma(db_path: str, collection_name: str, create_embeddings_fn=None):
    """
    Initialize Chroma DB.
    - If collection exists and has data → reuse it
    - If empty → create embeddings
    """
    global client, collection

    try:
        client = PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name=collection_name)

        # Check if collection already has data
        count = collection.count()

        if count == 0:
            logger.info("⚠️ Chroma collection is empty. Creating embeddings...")
            fn = create_embeddings_fn or create_embeddings
            collection = fn()
        else:
            logger.info(f"✅ Chroma collection loaded with {count} documents.")

    except Exception as e:
        logger.exception("Failed to initialize ChromaDB.")
        raise CustomException("Chroma initialization failed", e)




def get_collection():
    try:
        if collection is None:
            raise RuntimeError("Chroma collection not initialized.")
        return collection
    except Exception as e:
        logger.exception("Chroma collection access failed.")
        raise CustomException("Chroma collection is not available", e)