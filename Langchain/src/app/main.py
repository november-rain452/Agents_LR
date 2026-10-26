from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.core.config import GEMINIAPIKEY

# from langchain_core.messages import SystemMessage, HumanMessage

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=GEMINIAPIKEY)

# messages = [
#     SystemMessage(content="You are a concise programming tutor."),
#     HumanMessage(content="Explain the difference between a chain and an agent."),
# ]

parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {level} programming tutor."),
        ("human", "Expalin {topic} using a practical example."),
    ]
)

chain = prompt | model | parser


response = chain.invoke({"level": "beginner-friendly", "topic": "LCEL in langchain"})
print(response)
# messages = prompt.invoke({"level": "beginner-friendly", "topic": "LCEL in langchain"})
# response = model.invoke(messages)

# print(messages)
# print("-----")
# print(response.content[0]["text"])
