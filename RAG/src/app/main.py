from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from embedder import embedder

# initiating model
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
# initiating loader
loader = DirectoryLoader(
    path="C:\GitRepos\Agents_LR\RAG\src\data", glob="**/*.pdf", loader_cls=PyPDFLoader
)
# loading  docs
docs = loader.load()

# initiating chunker
text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)

# chunking
chunks = text_splitter.split_documents(docs)

# adding metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i


# setting up vectorstore

vector_storage = FAISS.from_documents(documents=chunks, embedding=embedder)

results = vector_storage.similarity_search(
    "What is the job requistion id of the profile most concerned with data modernization and migration?",
    k=2,
)

adds = []
for doc in results:
    adds.append(doc.page_content)
    print(doc.page_content)
    # print("*" * 25)

answer = model.invoke(
    "What is the job requistion id of the profile most concerned with data modernization and migration? Share the reason why."
    + adds[0]
)

print(answer.content)
