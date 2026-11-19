from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    recipient: str
    email_draft: str
    approved: bool
    status: str
    logs: Annotated[list[str], add_messages]


def generate_email(state: State):
    draft = (
        f"Hello {state['recipient']},\n\n"
        "I hope you are doing well. "
        "I am writing to follow up regarding our discussion.\n\n"
        "Best regards"
    )

    return {"email_draft": draft, "logs": ["Email drafted"]}


def approval_node(state: State):
    approved = interrupt(
        {
            "message": "Review this message.",
            "recipient": state["recipient"],
            "draft": state["email_draft"],
        }
    )
    return {"approved": approved, "logs": [f"Approval Gained: {approved}"]}


def send_email(state: State):
    print(f"Sending email to {state['recipient']} ")
    return {"status": "Email Sent", "logs": ["Email sent"]}


def reject_mail(state: State):
    print("Email Rejected")
    return {"status": "Email Rejected", "logs": [f"Approval Gained: False"]}


def routing_function(state: State):
    if state["approved"]:
        return "send"
    return "rejected"


builder = StateGraph(State)
builder.add_node("generate", generate_email)
builder.add_node("approval", approval_node)
builder.add_node("sender", send_email)
builder.add_node("rejecter", reject_mail)

builder.add_edge(START, "generate")
builder.add_edge("generate", "approval")
builder.add_conditional_edges(
    "approval", routing_function, {"send": "sender", "rejected": "rejecter"}
)
builder.add_edge("sender", END)
builder.add_edge("rejecter", END)


memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "email1"}}

ans = graph.invoke(
    {
        "recipient": "Taimur",
        "approved": None,
        "email_draft": "",
        "status": "",
        "logs": [],
    },
    config=config,
)


result = graph.invoke(
    Command(resume=False),
    config=config,
)
