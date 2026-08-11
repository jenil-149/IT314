import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def main() -> None:
    # Load the API key from the .env file.
    load_dotenv()

    # Fail fast if the key is missing.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    # Reuse the same Gemini model setup as the basic example.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )

    # Create a reusable prompt with placeholders for dynamic values.
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} to a {audience} in 3 clear bullet points."
    )

    # The | operator connects the template to the model into one chain.
    chain = prompt | llm
    response = chain.invoke({"topic": "LangChain", "audience": "beginner student"})
    print(response.content[0]["text"])


if __name__ == "__main__":
    main()
