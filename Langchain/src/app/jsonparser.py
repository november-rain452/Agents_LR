from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI


class Student(BaseModel):
    name: str
    age: int
    gender: Literal["male", "female", "NA"]


load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI bot. Your name is Antinous. You extract information from users and give structured reply, you figure out their name, age and gender and if you can't you return NA for that field. {format}",
        ),
        ("human", "Hello my name is {name}, {age} years old {fus}"),
    ]
)

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

parser = JsonOutputParser(pydantic_object=Student)
augprompt = prompt.partial(format=parser.get_format_instructions())

chain = augprompt | model | parser

response = chain.invoke({"name": "Adil", "age": 20, "fus": "a totally cool fella"})

print(response)
