"""
rag_pipeline.py
---------------
Handles everything related to the Retrieval-Augmented Generation pipeline:
  1. Load a PDF and split it into text chunks.
  2. Embed chunks and store them in ChromaDB (persisted locally).
  3. Retrieve the top-k most relevant chunks for a query.
  4. Generate an answer via an LLM using those chunks as context.

Dependencies:
  pip install langchain langchain-community langchain-openai chromadb pypdf openai
"""

import os
from typing import List, Tuple
print("RUNNING FILE:", __file__)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from utils import setup_logger, clean_text

logger = setup_logger()

# ---------------------------------------------------------------------------
# Configuration — override via environment variables or change defaults here
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME    = os.getenv("CHROMA_COLLECTION",   "support_kb")
#EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL",     "text-embedding-3-small")
LLM_MODEL          = os.getenv("LLM_MODEL",           "gpt-4o-mini")
TOP_K              = int(os.getenv("TOP_K", "2"))          # chunks to retrieve
SIMILARITY_THRESHOLD = 0.25
print("FINAL THRESHOLD:", SIMILARITY_THRESHOLD)  # minimum score

# Prompt template used for answer generation
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful customer support assistant.
Use ONLY the context below to answer the question.
If the context does not contain enough information, say "I don't know".

Context:
{context}

Question: {question}

Answer:""",
)


# ---------------------------------------------------------------------------
# 1. Load and chunk a PDF
# ---------------------------------------------------------------------------

def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """
    Load a PDF file and split it into overlapping text chunks.

    Returns a list of LangChain Document objects, each holding a chunk of text
    and metadata (source file, page number).
    """
    logger.info("Loading PDF: %s", pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # PyPDFLoader extracts text page-by-page
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    logger.info("Loaded %d page(s) from PDF.", len(pages))

    # Split pages into smaller, overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # characters per chunk
        chunk_overlap=100,    # overlap to preserve context across chunks
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Clean each chunk's text
    for doc in chunks:
        doc.page_content = clean_text(doc.page_content)

    logger.info("Split into %d chunk(s).", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# 2. Build / load the ChromaDB vector store
# ---------------------------------------------------------------------------

def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Embed chunks and persist them in ChromaDB.
    If the collection already exists, it is overwritten (safe for demos).
    """
    logger.info("Building vector store with %d chunks…", len(chunks))

    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info("Vector store persisted at '%s'.", CHROMA_PERSIST_DIR)
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an already-persisted ChromaDB collection (no re-embedding needed).
    Call this after build_vector_store() has been run at least once.
    """
    logger.info("Loading existing vector store from '%s'…", CHROMA_PERSIST_DIR)
    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


# ---------------------------------------------------------------------------
# 3. Retrieve relevant chunks
# ---------------------------------------------------------------------------

def retrieve_docs(vector_store: Chroma, query: str) -> Tuple[List[Document], float]:
    """
    Run a similarity search against the vector store.

    Returns:
        docs        : List of retrieved Document chunks (may be empty).
        best_score  : Highest similarity score found (0.0 if nothing retrieved).

    similarity_search_with_score returns distance (lower = better)
    We convert it to similarity (0–1) using: similarity = 1 / (1 + distance)
    """
    logger.info("Retrieving top-%d docs for query: '%s'", TOP_K, query)

    results = vector_store.similarity_search_with_score(query, k=TOP_K)

    if not results:
        logger.warning("No documents retrieved.")
        return [], 0.0

    # Convert distance → similarity
    docs_with_scores = []
    for doc, distance in results:
        similarity = 1 / (1 + distance)
        docs_with_scores.append((doc, similarity))

    # Filter based on similarity
    filtered = [(doc, score) for doc, score in docs_with_scores if score >= SIMILARITY_THRESHOLD]

    if not filtered:
        logger.warning("Using top results despite low similarity.")
        docs = [doc for doc, _ in docs_with_scores]
        best_score = max(score for _, score in docs_with_scores)
        return docs, best_score

    docs = [doc for doc, _ in filtered]
    best_score = max(score for _, score in filtered)

    logger.info("Retrieved %d doc(s) above threshold. Best score: %.2f", len(docs), best_score)

    return docs, best_score
        


# ---------------------------------------------------------------------------
# 4. Generate an answer from the LLM
# ---------------------------------------------------------------------------

def generate_answer(query: str, docs: List[Document]) -> str:
    """
    Concatenate retrieved chunks into a context string and ask the LLM to
    answer the user's question.  Returns the LLM's text response.
    """
    if not docs:
        return "I don't know — no relevant documents were found in the knowledge base."

    # Build context from retrieved chunks
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Format the prompt
    prompt_text = QA_PROMPT.format(context=context, question=query)

    model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    logger.info(f"Sending prompt to LLM (Groq {model_name})...")

    llm = ChatGroq(model=model_name,temperature=0)
    response = llm.invoke(prompt_text)

    answer = response.content.strip()
    logger.info("LLM response received (%d chars).", len(answer))
    return answer
