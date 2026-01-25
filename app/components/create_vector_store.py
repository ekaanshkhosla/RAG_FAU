from app.components.create_chunks import create_chunks
import threading
from app.config.config import DB_NAME, MODEL, openai, embedding_model, RETRIEVAL_K
from chromadb import PersistentClient
from app.config.config import DB_NAME, collection_name, openai, embedding_model
from pathlib import Path
from app.utils.logger import get_logger

logger = get_logger(__name__)

def create_embeddings():

    """

    Creates embeddings and stores them in Chroma.

    Assumes it is called ONLY once (inside _collection_lock).

    """

    logger.info("Creating embeddings...")
    chunks = create_chunks()
    chroma = PersistentClient(path=DB_NAME)

    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)


    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    logger.info(f"Vectorstore created with {collection.count()} documents")

    return collection 