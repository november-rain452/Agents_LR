from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
import operator
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from test_checkpoints import view_checkpoint_history

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


class Evaluate(BaseModel):
    quality_ok: bool = Field(
        description="Field to denote if the quality is upto par or not"
    )


evaluator = llm.with_structured_output(Evaluate)


class State(TypedDict):
    question: str
    answer: str
    attempts: int
    quality_ok: bool
    logs: Annotated[list[str], operator.add]


def generate_improve_node(state: State):
    attempt = state["attempts"] + 1

    if attempt == 1:
        prompt = f"""You are an answering agent. Answer the given question adequately. Question : {state['question']}"""
    if attempt > 1:
        prompt = f"""Improve the answer if quality isnt okay. Quality okay : {state['quality_ok']}
         Question : {state['question']}.    
          Previous answer : {state["answer"]} """

    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "attempts": attempt,
        "logs": [f"Attempt :{attempt}"],
    }


def evaluate_node(state: State):
    prompt = f"""You are an evaluator. Return true or false based on the quality of the text being appropriate or not.
    Question : {state['question']}.
    Answer : {state['answer']}."""
    ans = evaluator.invoke(prompt)
    return {
        "quality_ok": ans.quality_ok,
        "logs": [f"Evaluated answer as : {ans.quality_ok}"],
    }


def choose_step(state: State):
    if state["attempts"] > 2:
        return "finish"
    if state["quality_ok"]:
        return "finish"

    return "retry"


builder = StateGraph(State)

builder.add_node("gen", generate_improve_node)
builder.add_node("evaluate", evaluate_node)

builder.add_edge(START, "gen")
builder.add_edge("gen", "evaluate")
builder.add_conditional_edges("evaluate", choose_step, {"finish": END, "retry": "gen"})

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "conv1"}}
config_2 = {"configurable": {"thread_id": "conv2"}}

ans = graph.invoke(
    {
        "question": "What is RAG in Gen AI?",
        "answer": "",
        "attempts": 0,
        "quality_ok": False,
        "logs": [],
    },
    config=config,
)

ans2 = graph.invoke(
    {
        "question": "What is Fine Tuning in Gen AI?",
        "answer": "",
        "attempts": 0,
        "quality_ok": False,
        "logs": [],
    },
    config=config_2,
)


# print(ans)
history = graph.get_state_history(config)
state_snapshot = graph.get_state(config)

history2 = graph.get_state_history(config_2)

view_checkpoint_history(history)
view_checkpoint_history(history2)
