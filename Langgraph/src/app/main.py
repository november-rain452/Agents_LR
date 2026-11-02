from langgraph.graph import START, StateGraph, END
from dotenv import load_dotenv
from typing import Annotated
import operator
from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    category: str
    answer: str
    logs: Annotated[list[str], operator.add]


def classify_question_node(state: State):
    question = state["question"].lower()

    if any(word in question for word in ["java", "python", "sql", "c++"]):
        category = "technical"
    else:
        category = "general"

    return {"category": category, "logs": [f"Classified as {category}"]}


def technical_answer_node(state: State):
    return {
        "answer": f"Technical question recieved, Question : {state['question']}",
        "logs": ["Technical route executed"],
    }


def general_answer_node(state: State):
    return {
        "answer": f"General question recieved, Question : {state['question']}",
        "logs": ["General route executed"],
    }


def route_question_node(state: State):
    return state["category"]


builder = StateGraph(State)

builder.add_node("classify", classify_question_node)
builder.add_node("technical_n", technical_answer_node)
builder.add_node("general_n", general_answer_node)


builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route_question_node,
    {"technical": "technical_n", "general": "general_n"},
)
builder.add_edge("technical_n", END)
builder.add_edge("general_n", END)

graph = builder.compile()


response = graph.invoke({"question": "Whats a API?", "logs": []})
print(response)
