from app.config.config import MODEL
from litellm import completion
from app.components.get_relevant_chunks import get_relevant_chunks
from app.utils.logger import get_logger
from app.utils.custom_exception import CustomException


logger = get_logger(__name__)


def rewrite_query(question: str) -> str:
    try:
        logger.info("Rewriting query for better retrieval....")

        message = f"""
You are in a conversation with a user, answering questions about the study advisory for the Department of Data Science at FAU Erlangen.
You are about to look up information in a Knowledge Base to answer the user's question.

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short, specific question most likely to surface content relevant to the Data Science study programs at FAU Erlangen.
Focus on concrete details (e.g., admissions, course structure, requirements, deadlines, modules).
Don't mention FAU Erlangen unless it's necessary to clarify context.
IMPORTANT: Respond ONLY with the knowledge base search query, nothing else.
"""
        response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
        return response.choices[0].message.content
    except Exception as e:
        logger.exception("Failed to rewrite query.")
        raise CustomException("Query rewrite failed", e)




def make_rag_messages(question: str, chunks):
    try:
        logger.info("Giving answer based on retrieved context...")

        SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the study advisory for the Department of Data Science at FAU Erlangen.
You are chatting with a user about the Department of Data Science and its study programs.
Your answer will be evaluated for accuracy, relevance, and completeness, so make sure it only answers the question and fully answers it.
If you don’t know the answer, say so.

For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user’s question:
{context}

With this context, please answer the user’s question. Be accurate, relevant, and complete.
"""
        context = "\n\n".join(
            f"Extract from {chunk.metadata.get('source', 'unknown')}:\n{chunk.page_content}"
            for chunk in chunks
        )
        system_prompt = SYSTEM_PROMPT.format(context=context)
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    except Exception as e:
        logger.exception("Failed to build RAG messages.")
        raise CustomException("Failed to construct RAG prompt/messages", e)





def answer_question(question: str) -> str:
    """
    Answer a question using RAG and return the answer.
    """
    try:
        query = rewrite_query(question)
        relevant_chunks = get_relevant_chunks(query)
        messages = make_rag_messages(question, relevant_chunks)
        response = completion(model=MODEL, messages=messages)
        return response.choices[0].message.content
    except CustomException:
        # already wrapped, bubble up
        raise
    except Exception as e:
        logger.exception("Failed to answer question.")
        raise CustomException("Answer generation failed", e)