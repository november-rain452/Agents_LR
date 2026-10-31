from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader(
    path="C:\GitRepos\Agents_LR\RAG\src\data", glob="**/*.pdf", loader_cls=PyPDFLoader
)


# loading  docs and chunking
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i


def give_chunks():
    return chunks
