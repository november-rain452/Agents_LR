from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from src.app.embedder import rag_tool

# initiating model
load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

agent = create_agent(
    model=model,
    tools=[rag_tool],
    system_prompt="You are a crawler, your job is to hunt for info, if you are unclear about the information seek it through the internal knowledge base.",
)


response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the job requistion id of the profile most concerned with data modernization and migration?",
            }
        ]
    }
)

print(response["messages"][-1].content[0]["text"])
