from app.components.create_chunks import create_chunks
import threading
from app.config.config import DB_NAME, MODEL, openai, embedding_model, RETRIEVAL_K
from chromadb import PersistentClient
from app.config.config import DB_NAME, collection_name, openai, embedding_model
from pathlib import Path
from app.utils.logger import get_logger
from app.utils.custom_exception import CustomException


logger = get_logger(__name__)

def create_embeddings():
    """
    Creates embeddings and stores them in Chroma.
    """
    try:
        logger.info("Creating embeddings...")

        chunks = create_chunks()

        chroma = PersistentClient(path=DB_NAME)

        # Reset collection if exists
        try:
            existing = [c.name for c in chroma.list_collections()]
            if collection_name in existing:
                chroma.delete_collection(collection_name)
        except Exception as e:
            logger.exception("Failed while checking/deleting existing Chroma collection.")
            raise CustomException("Failed to reset Chroma collection", e)

        texts = [chunk.page_content for chunk in chunks]
        metas = [chunk.metadata for chunk in chunks]
        ids = [str(i) for i in range(len(chunks))]

        if not texts:
            raise ValueError("No chunks found to embed")

        emb = openai.embeddings.create(model=embedding_model, input=texts).data
        vectors = [e.embedding for e in emb]

        collection = chroma.get_or_create_collection(collection_name)
        collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

        logger.info(f"Vectorstore created with {collection.count()} documents")
        return collection

    except CustomException:
        raise
    except Exception as e:
        logger.exception("Failed to create vector store embeddings.")
        raise CustomException("Vector store creation failed", e)