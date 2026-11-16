import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


class State(TypedDict):
    message: Sequence[BaseMessage]
    logs: Annotated[list[str], operator.add]


def question_node(state: State):
    print("Whats your question?")
    question = [HumanMessage(content=input())]
    return {"message": question, "logs": [f"Asking the question: {question}"]}


def answer_node(state: State):
    answer = llm.invoke(state["message"])
    return {"message": [answer], "logs": ["LLM answered the question"]}


builder = StateGraph(State)
builder.add_node("ques", question_node)
builder.add_node("ans", answer_node)

builder.add_edge(START, "ques")
builder.add_edge("ques", "ans")
builder.add_edge("ans", END)

conn = sqlite3.connect(
    r"C:/GitRepos/Agents_LR/Langgraph/src/app/memory/DB/agent_checkpoints.sqlite",
    check_same_thread=False,
)
memory = SqliteSaver(conn)

graph = builder.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "id1"}}
ans = graph.invoke({"message": [], "logs": []}, config=config)
print(ans)
conn.close()
