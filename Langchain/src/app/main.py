from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from src.core.config import GEMINIAPIKEY

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=GEMINIAPIKEY)

# messages = [
#     SystemMessage(content="You are a concise programming tutor."),
#     HumanMessage(content="Explain the difference between a chain and an agent."),
# ]
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {level} programming tutor."),
        ("human", "Expalin {topic} using a practical example."),
    ]
)
messages = prompt.invoke({"level": "beginner-friendly", "topic": "LCEL in langchain"})
response = model.invoke(messages)

print(messages)
print("-----")
print(response.content[0]["text"])
