from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)


response = model.invoke("Explain Rag in three sentences")

print(response.content[0]["text"])
