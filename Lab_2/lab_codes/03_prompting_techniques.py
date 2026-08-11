import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class TicketDetails(BaseModel):
    # Pydantic schema used to force structured output from the model.
    category: Literal["billing", "technical", "account", "other"] = Field(
        description="The best matching support category."
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent the request appears to be."
    )
    summary: str = Field(description="A short summary of the request.")
    recommended_reply: str = Field(description="A concise response the support team could send.")


TASK_MESSAGE = """I can't log into my account after resetting my password. The page says 'invalid token'."""


def get_llm() -> ChatGoogleGenerativeAI:
    # Shared helper so every demo uses the same Gemini setup.
    load_dotenv()

    # Keep the API key in the environment instead of hardcoding it.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )


def print_heading(title: str) -> None:
    # Make each technique easy to spot in the console output.
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def zero_shot_prompting(llm: ChatGoogleGenerativeAI) -> None:
    # No examples are provided here; the model must infer the category directly.
    print_heading("Zero-shot prompting")
    prompt = (
        "Classify this customer message into one of these categories: billing, technical, account, other.\n"
        f"Message: {TASK_MESSAGE}\n"
        "Return only the category name."
    )
    response = llm.invoke(prompt)
    print(response.content[0]["text"])


def one_shot_prompting(llm: ChatGoogleGenerativeAI) -> None:
    # One example is shown to guide the model's answer style.
    print_heading("One-shot prompting")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify customer messages into one of these categories: billing, technical, account, other.",
            ),
            (
                "human",
                "Example:\nMessage: I was charged twice for my subscription.\nCategory: billing",
            ),
            ("human", "Message: {message}\nCategory:"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"message": TASK_MESSAGE})
    print(response.content[0]["text"])


def few_shot_prompting(llm: ChatGoogleGenerativeAI) -> None:
    # Multiple examples usually improve consistency for simple classification tasks.
    print_heading("Few-shot prompting")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify customer messages into one of these categories: billing, technical, account, other.",
            ),
            ("human", "Message: I was charged twice for my subscription.\nCategory: billing"),
            ("human", "Message: The app crashes when I tap the save button.\nCategory: technical"),
            ("human", "Message: I need to update my email address.\nCategory: account"),
            ("human", "Message: {message}\nCategory:"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"message": TASK_MESSAGE})
    print(response.content[0]["text"])


def chain_of_thought_prompting(llm: ChatGoogleGenerativeAI) -> None:
    # Ask for a step-by-step solution for a small reasoning problem.
    print_heading("Chain-of-thought prompting")
    prompt = (
        "Solve this problem step by step, then give the final answer at the end.\n"
        "A notebook costs $3 and a pen costs $2. If you buy 4 notebooks and 5 pens, how much do you pay?"
    )
    response = llm.invoke(prompt)
    print(response.content[0]["text"])


def role_prompting(llm: ChatGoogleGenerativeAI) -> None:
    # A role in the system message changes the tone and depth of the response.
    print_heading("Role prompting")
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a senior software engineer.") ,
            HumanMessage(content="Review this idea: keep all validation logic inside one function. What would you recommend?"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({})
    print(response.content[0]["text"])


def output_formatting(llm: ChatGoogleGenerativeAI) -> None:
    # Structured output makes the result easier for application code to consume.
    print_heading("Output formatting")
    structured_llm = llm.with_structured_output(TicketDetails)
    prompt = ChatPromptTemplate.from_template(
        """
        Extract support ticket details from this customer message.
        Message: {message}
        """
    )
    chain = prompt | structured_llm
    result = chain.invoke({"message": TASK_MESSAGE})
    print(result.model_dump())


def reducing_hallucinations(llm: ChatGoogleGenerativeAI) -> None:
    # Limit the model to a known context and force an "I don't know" fallback.
    print_heading("Reducing hallucinations")
    context = (
        "Project Atlas ships on Fridays. The support email is support@atlas.dev. "
        "The office hours are 9 AM to 5 PM UTC."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer only using the provided context. If the answer is not in the context, say: I don't know based on the provided information.",
            ),
            ("human", "Context: {context}\nQuestion: {question}"),
        ]
    )
    chain = prompt | llm

    answerable = chain.invoke({"context": context, "question": "What is the support email?"})
    unknown = chain.invoke({"context": context, "question": "Who is the CEO of Project Atlas?"})

    print("Answerable question:")
    print(answerable.content[0]["text"])
    print("\nUnanswerable question:")
    print(unknown.content[0]["text"])


def main() -> None:
    # Run every prompting example in sequence.
    llm = get_llm()

    zero_shot_prompting(llm)
    one_shot_prompting(llm)
    few_shot_prompting(llm)
    chain_of_thought_prompting(llm)
    role_prompting(llm)
    output_formatting(llm)
    reducing_hallucinations(llm)


if __name__ == "__main__":
    main()
