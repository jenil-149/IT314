import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    """Initializes and returns the LangChain Google GenAI model."""
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not google_api_key:
        print(
            "ERROR: GOOGLE_API_KEY environment variable is not set.\n"
            "Set it with: export GOOGLE_API_KEY='your-key-here'\n"
            "Or ensure it is placed in your .env file."
        )
        sys.exit(1)

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",  
        temperature=0.3,           
        google_api_key=google_api_key,
    )