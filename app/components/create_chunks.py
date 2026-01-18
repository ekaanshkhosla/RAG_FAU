from litellm import completion
from app.config.config import KNOWLEDGE_BASE_PATH
from app.config.config import AVERAGE_CHUNK_SIZE
from app.config.config import MODEL
from tqdm import tqdm
from app.components.schemas import Chunks



def fetch_documents():
    """Loads all markdown files from the transformed_data_llm_cleaned folder"""

    documents = []

    for file in KNOWLEDGE_BASE_PATH.rglob("*.md"):
        with open(file, "r", encoding="utf-8") as f:
            documents.append({
                "type": "Knowledge base for Masters in Data Science at FAU",           # any label you want
                "source": file.as_posix(),
                "text": f.read()
            })

    print(f"Loaded {len(documents)} documents from {KNOWLEDGE_BASE_PATH}")
    return documents




def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a Knowledge Base.

The document contains public information about the Department of Data Science at FAU.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}
The content was collected from public websites and/or public PDFs and converted to markdown.

A chatbot will use these chunks to answer questions about the department, its programs, courses,
and related information.

Your task:
- Divide the document into coherent chunks that are useful for retrieval.
- Make sure the **entire** document is covered in the chunks; do not leave anything out.
- This document will probably be split into about {how_many} chunks, but you may use more or fewer as appropriate.
- There should be overlap between neighboring chunks (typically ~25% overlap or about 50 words)
  so that important context appears in multiple chunks.
- Preserve important markdown structure such as headings, bullet lists, tables, and links inside each chunk.
- Do **not** add new facts; only use the information that appears in the document.

For each chunk, provide:
1. A short headline for the chunk.
2. A brief summary of the chunk.
3. The original text of the chunk (from the document, with markdown preserved).

Together, your chunks should represent the entire document with appropriate overlap.

Here is the document in markdown:

{document["text"]}

Respond with the chunks.
"""



def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]




def process_document(document):
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]




def create_chunks():
    documents = fetch_documents()
    chunks = []
    for doc in tqdm(documents):
        chunks.extend(process_document(doc))
    return chunks


