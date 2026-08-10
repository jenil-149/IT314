import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=api_key
)

result = model.invoke("What is the full form of DAIICT?")

print(result.content)