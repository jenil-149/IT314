import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI


def main() -> None:
    # Load the API key from the local environment file.
    load_dotenv()

    # Keep credentials out of source code.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    # Create the shared Gemini model used by both chain stages.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=api_key,
        temperature=0.4,
    )

    # Stage 1: turn a topic into a short beginner-friendly explanation.
    prompt_1 = ChatPromptTemplate.from_template(
        "Write a short technical explanation of this topic for a beginner: {topic}"
    )
    # Stage 2: reuse the explanation and rewrite it as a LinkedIn post.
    prompt_2 = ChatPromptTemplate.from_template(
        "Turn the explanation below into a concise LinkedIn post with a friendly professional tone.\n\n"
        "Topic: {topic}\n"
        "Explanation: {explanation}\n\n"
        "Keep it under 120 words and add 2 relevant hashtags."
    )

    # Parse the first LLM response into plain text before passing it onward.
    explanation_chain = prompt_1 | llm | StrOutputParser()

    # Build the full two-step LCEL pipeline.
    full_chain = (
        RunnablePassthrough.assign(explanation=explanation_chain)
        | prompt_2
        | llm
        | StrOutputParser()
    )

    # Invoke the chain with a single topic input.
    topic = "LangChain prompt templates"
    result = full_chain.invoke({"topic": topic})
    print(result)


if __name__ == "__main__":
    main()
