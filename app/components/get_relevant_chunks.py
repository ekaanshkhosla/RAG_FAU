from app.components.create_chunks import create_chunks
from litellm import completion
import threading
from app.config.config import DB_NAME, MODEL, openai, embedding_model, RETRIEVAL_K
from chromadb import PersistentClient
from app.config.config import DB_NAME, collection_name, openai, embedding_model
from app.components.schemas import Result, RankOrder
from app.components.chroma_store import get_collection
from app.utils.logger import get_logger

logger = get_logger(__name__)


def rerank(question, chunks):

    logger.info("Ordering the chunks by relevance to the question asked...")

    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]




def fetch_context_unranked(query):
    collection = get_collection()
    logger.info("Querying chroma DB...")
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks




def get_relevant_chunks(question):
    logger.info("Getting relevant chunks from chroma DB...")
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    chunks = fetch_context_unranked(query)
    ranked_chunks =  rerank(question, chunks)
    return ranked_chunks

