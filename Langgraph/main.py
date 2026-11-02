from langgraph.graph import START, StateGraph, END
from dotenv import load_dotenv
from typing import Annotated
import operator
from pydantic import BaseModel


class State(BaseModel):
    question: str
    category: str
    answer: str
    logs: Annotated[list[str], operator.add]
