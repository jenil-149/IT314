import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def main() -> None:
    # Load environment variables from .env so the API key stays out of the code.
    load_dotenv()

    # Read the Google API key safely from the environment.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    # Create a simple Gemini chat model instance.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )

    # Send a plain text prompt directly to the model.
    prompt = "Write one friendly sentence explaining what LangChain is."
    response = llm.invoke(prompt)
    print(response.content[0]["text"])


if __name__ == "__main__":
    main()
