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
