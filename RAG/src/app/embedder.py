from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.documents.base import Document
from langchain_core.tools import tool
from src.core.loader import chunks

load_dotenv()

embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

vector_store = FAISS.from_documents(embedding=embedder, documents=chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})


def create_embeddings_query(query: str) -> list:

    vector = embedder.embed_query(query)
    return vector


def create_embeddings_documents(docs) -> list:
    vectors = embedder.embed_documents(docs)
    return vectors


@tool
def rag_tool(query: str) -> list[Document]:
    """Retrieve information from the vector storage relevant to the query."""
    responses = retriever.invoke(query)
    if not responses:
        return "No relevant information found in the knowledge base."

    formatted = "\n\n".join(
        f"Source : {response.metadata.get("source","Unavailable")} \n Content: {response.page_content}"
        for response in responses
    )
    return formatted
