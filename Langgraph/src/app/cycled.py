import operator
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated


class State(TypedDict):
    question: str
    answer: str
    attempts: int
    feedback: str
    quality_ok: bool
    logs: Annotated[list[str], operator.add]


def generate_or_improve(state: State):

    attempts = state["attempts"] + 1

    if attempts == 1:
        answer = "RAG is retrieval augmented generation."
    elif state["feedback"] == "Too Short":
        answer = "RAG is retrieval augmented generation. It is used when we need the LLM to work with fresh data. It is also useful to get around shabby context window limits."

    return {
        "answer": answer,
        "attempts": attempts,
        "logs": [f"Attempt : {attempts}. Answer attempted "],
    }


def evaluate_answer(state: State):
    quality = len(state["answer"]) > 60
    if not quality:
        feedback = "Too Short"
    else:
        feedback = "Looks good"
    return {
        "quality_ok": quality,
        "feedback": feedback,
        "logs": [
            f"Evaluated the quality and marked it {"Passed" if quality else "Failed"}"
        ],
    }


def decide_next_step(state: State):
    if state["quality_ok"]:
        return "finish"

    if state["attempts"] >= 2:
        return "finish"

    return "retry"


builder = StateGraph(State)

builder.add_node("gen_improve", generate_or_improve)
builder.add_node("evaluate", evaluate_answer)

builder.add_edge(START, "gen_improve")
builder.add_edge("gen_improve", "evaluate")
builder.add_conditional_edges(
    "evaluate", decide_next_step, {"retry": "gen_improve", "finish": END}
)

graph = builder.compile()

result = graph.invoke(
    {
        "question": "What is rag",
        "logs": [],
        "attempts": 0,
        "quality_ok": False,
        "answer": "",
        "feedback": "",
    }
)
print(result)
