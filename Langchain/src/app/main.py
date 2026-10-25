from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.config import GEMINIAPIKEY

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=GEMINIAPIKEY)

messages = [
    SystemMessage(content="You are a concise programming tutor."),
    HumanMessage(content="Explain the difference between a chain and an agent."),
]

response = model.invoke(messages)

print(response.content[0]["text"])
