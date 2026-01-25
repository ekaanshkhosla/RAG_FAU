from app.components.create_chunks import create_chunks
from litellm import completion
import threading
from app.config.config import DB_NAME, MODEL, openai, embedding_model, RETRIEVAL_K
from chromadb import PersistentClient
from app.config.config import DB_NAME, collection_name, openai, embedding_model
from app.components.schemas import Result, RankOrder
from app.components.chroma_store import get_collection
from app.utils.logger import get_logger
from app.utils.custom_exception import CustomException


logger = get_logger(__name__)


def rerank(question: str, chunks):
    try:
        logger.info("Ordering the chunks by relevance to the question asked...")

        system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
        user_prompt = f"The user has asked the following question:\n\n{question}\n\n"
        user_prompt += "Order all the chunks of text by relevance to the question, from most relevant to least relevant.\n\n"
        user_prompt += "Here are the chunks:\n\n"
        for index, chunk in enumerate(chunks):
            user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
        user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = completion(model=MODEL, messages=messages, response_format=RankOrder)

        reply = response.choices[0].message.content
        order = RankOrder.model_validate_json(reply).order

        # Defensive: ensure order is valid
        if not order or any(i < 1 or i > len(chunks) for i in order):
            raise ValueError("Invalid rerank order returned by model")

        return [chunks[i - 1] for i in order]
    except Exception as e:
        logger.exception("Reranking failed.")
        raise CustomException("Chunk reranking failed", e)





def fetch_context_unranked(query_embedding):
    try:
        collection = get_collection()
        logger.info("Querying Chroma DB...")

        results = collection.query(query_embeddings=[query_embedding], n_results=RETRIEVAL_K)

        chunks = []
        for doc, meta in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]):
            chunks.append(Result(page_content=doc, metadata=meta))
        return chunks
    except Exception as e:
        logger.exception("Chroma query failed.")
        raise CustomException("Failed to query Chroma DB", e)





def get_relevant_chunks(question: str):
    try:
        logger.info("Getting relevant chunks from Chroma DB...")

        # Embed the query
        query_embedding = openai.embeddings.create(
            model=embedding_model,
            input=[question],
        ).data[0].embedding

        chunks = fetch_context_unranked(query_embedding)

        # If nothing retrieved, return empty list (or raise if you prefer)
        if not chunks:
            logger.warning("No chunks retrieved from Chroma.")
            return []

        ranked_chunks = rerank(question, chunks)
        return ranked_chunks
    except CustomException:
        raise
    except Exception as e:
        logger.exception("Failed while retrieving relevant chunks.")
        raise CustomException("Retrieval pipeline failed", e)