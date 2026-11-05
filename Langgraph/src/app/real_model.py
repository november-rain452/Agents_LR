from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
import operator
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)


class AgentState(TypedDict):
    question: str
    answer: str
    attempts: int
    feedback: str
    quality_ok: bool
    logs: Annotated[list[str], operator.add]


class Evaluation(BaseModel):
    quality_ok: bool = Field(
        description="Whether the answer adequately answers the question."
    )
    feedback: str = Field(description="Specific improvements needed, or why it passed")


evaluator = llm.with_structured_output(Evaluation)


def generate_or_improve(state: AgentState):
    attempts = state["attempts"] + 1

    if attempts == 1:
        prompt = f"""Answer the following questions clearly and concisely.
        Question : {state['question']}
        Provide details so that the answer is useful."""
    else:
        prompt = f"""Improve the answer based on the evaluator's feedback. 
        Original Question : {state['question']}
        
        Previous Answer : {state['answer']}

        Feedback : {state['feedback']}
        
        Write a revised answer that addresses the feedback
        and answers the original question accurately."""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "attempts": attempts,
        "logs": [f"Generated answer for attempt : {attempts}"],
    }


def evaluate_answer(state: AgentState):
    prompt = f""" You are an evaluation agent. Provide feedback and evaluate if the answer accurately fits the question based on the rules.
        
                Question : {state['question']}

                The answer : {state['answer']}
            
            Evaluation criteria:    
            - Does it answer the question accurately?
            - Does it answer address the question?
            - Does it provide enough details to explain the essential concepts?
            - Is it clear and easy to understand?
            
            
            Set quality_ok to be true only if the answer meets these criteria.
            If the answer doesn't meet it set it to false and write a descriptive feedback so that the answer may be fixed.
            
            Otherwise, explain  briefly why it's adequate."""

    evaluation = evaluator.invoke(prompt)

    return {
        "quality_ok": evaluation.quality_ok,
        "feedback": evaluation.feedback,
        "logs": [f"Evaluation : {'Passed' if evaluation.quality_ok else 'Failed'}"],
    }


MAX_ATTEMPTS = 3


def decide_next_step(state: AgentState):
    if state["quality_ok"]:
        return "finish"

    if state["attempts"] >= MAX_ATTEMPTS:
        return "finish"

    return "retry"
