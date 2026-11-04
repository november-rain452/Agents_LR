from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
import operator

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)


class AgentState(TypedDict):
    question: str
    answer: str
    attempts: int
    feedback: str
    quality_ok: bool
    logs: Annotated[list[str], operator.add]
