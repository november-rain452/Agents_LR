# using langchain without the magic shortcuts

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.core.config import GEMINIAPIKEY

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=GEMINIAPIKEY)


parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {level} programming tutor."),
        ("human", "Expalin {topic} using a practical example briefly."),
    ]
)

msg = prompt.invoke({"level": "medium-level", "topic": "langchain vs langgraph"})

msg = model.invoke(msg)

msg = parser.invoke(msg)

print(msg)
