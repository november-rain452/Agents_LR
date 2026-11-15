import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-lite")


class State(TypedDict):
    message = Sequence[BaseMessage]
    logs = Annotated[str, operator.add]


def question_node(state: State):
    print("Whats your question?")
    question = [BaseMessage(content=input())]
    return {"message": question, "logs": [f"Asking the question: {question}"]}


def answer_node(state: State):
    answer = llm.invoke(state["message"])
    return {"message": answer.content}
