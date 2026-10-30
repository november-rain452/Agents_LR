from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")


def create_embeddings_query(query: str) -> list:

    vector = embedder.embed_query(query)
    return vector


def create_embeddings_documents(docs) -> list:
    vectors = embedder.embed_documents(docs)
    return vectors
