"""
Issue Triage Assistant
-----------------------
Classifies an incoming issue message into one of four categories and
generates a category-specific reply, using LangChain + Google Gemini.

Categories:
    bug_report          - something is broken
    feature_request     - a suggestion for new/improved functionality
    documentation_issue - docs are missing/unclear/incorrect
    question             - a general how-do-I / clarification question
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Setup

load_dotenv()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from .env

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    raise EnvironmentError(
        "No API key found. Add GEMINI_API_KEY=your_key to a .env file."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.3,
)

VALID_CATEGORIES = {
    "bug_report",
    "feature_request",
    "documentation_issue",
    "question",
}

# 2. Classification chain

classification_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an issue triage classifier for a software product team. "
            "Read the user's message and classify it into EXACTLY ONE of these without any other extra text"
            "four categories:\n"
            "- bug_report: something in the software is broken or behaving incorrectly\n"
            "- feature_request: a suggestion for new or improved functionality\n"
            "- documentation_issue: the docs/README/guide are missing, unclear, or incorrect\n"
            "- question: a general how-do-I / clarification question that isn't a bug or a request\n\n"
            "Respond with ONLY the category name, in lowercase, with underscores, "
            "and nothing else. No punctuation, no explanation.",
        ),
        ("human", "{message}"),
    ]
)

classification_chain = classification_prompt | llm | StrOutputParser()


def classify_message(message: str) -> str:
    """Runs the classification chain and normalizes/validates the result."""
    raw_result = classification_chain.invoke({"message": message})
    category = raw_result.strip().lower().replace(" ", "_").replace("-", "_")

    if category not in VALID_CATEGORIES:
        # Fallback: if the model returns anything unexpected, default to
        # 'question' so the pipeline never crashes on a bad classification.
        category = "question"

    return category


# 3. Category-specific reply prompts (the routing step)
REPLY_SYSTEM_PROMPTS = {
    "bug_report": (
        "You are a support engineer replying to a bug report. "
        "Acknowledge the bug clearly. If the message is missing reproduction "
        "steps, environment/version details, or expected vs. actual behaviour, "
        "ask for whichever of these are missing. Be concise, professional, and "
        "empathetic. Do not say 'thanks for reaching out' generically — engage "
        "with the specific bug described."
    ),
    "feature_request": (
        "You are a product team member replying to a feature request. "
        "Thank the user specifically for the suggestion, ask what problem this "
        "feature would solve or how they would use it day-to-day, and let them "
        "know the team will consider it for the roadmap. Be warm but concise."
    ),
    "documentation_issue": (
        "You are a technical writer replying to a documentation complaint. "
        "Acknowledge the specific documentation gap described. Ask which "
        "page/section was unclear or incorrect if not already stated. Thank "
        "them for helping improve the docs. Be concise and appreciative."
    ),
    "question": (
        "You are a helpful support agent answering a general how-do-I or "
        "clarification question. Try to answer it directly and helpfully "
        "based on the information given. If you genuinely cannot answer it "
        "from the message alone, point them to where they could find the "
        "answer (e.g. docs, FAQ, or support channel). Be concise and direct."
    ),
}

reply_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "{system_instructions}\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- MAXIMUM LENGTH: 5 -8 sentences. Be ruthless about conciseness.\n"
            "- NO GREETINGS: Do not start with 'Hi', 'Hello', or 'Thanks'.\n"
            "- NO SIGN-OFFS: Do not include any signatures, sign-offs, or placeholders like [Your Name].\n"
            "- PLAIN TEXT ONLY: Output strictly in plain text. Do not use Markdown formatting like ** or *."
        ),
        ("human", "Issue message:\n{message}\n\nWrite the reply."),
    ]
)

reply_chain = reply_prompt | llm | StrOutputParser()


def generate_reply(message: str, category: str) -> str:
    """Routes to the category-specific prompt and generates a reply."""
    system_instructions = REPLY_SYSTEM_PROMPTS[category]
    reply = reply_chain.invoke(
        {"system_instructions": system_instructions, "message": message}
    )
    return reply.strip()


# 4. Pipeline entrypoint

def run_pipeline(message: str) -> None:
    category = classify_message(message)
    reply = generate_reply(message, category)

    print("=" * 70)
    print("INPUT MESSAGE:")
    print(message)
    print("-" * 70)
    print(f"CLASSIFICATION RESULT: {category}")
    print("-" * 70)
    print("GENERATED REPLY:")
    print(reply)
    print("=" * 70)
    print()


# 5. Main - interactive terminal loop
def main():
    print("Issue Triage Assistant")
    print("Type an issue message and press Enter. Type 'exit' to quit.\n")

    while True:
        message = input("Enter issue message: ").strip()
        if message.lower() in ("exit", "quit","end"):
            print("Goodbye!")
            break
        if not message:
            print("Please enter a non-empty message.\n")
            continue

        run_pipeline(message)


if __name__ == "__main__":
    main()