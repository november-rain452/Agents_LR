from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

loader = DirectoryLoader(
    path="C:\GitRepos\Agents_LR\RAG\src\data", glob="**/*.pdf", loader_cls=PyPDFLoader
)

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)

chunks = []

chunks = text_splitter.split_documents(docs)

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i

print(chunks[0].metadata)
