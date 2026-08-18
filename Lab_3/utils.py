import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# Import the prompts from your new prompts.py file
from prompts import (
    STAGE_1_TEMPLATE,
    STAGE_2_TEMPLATE,
    STAGE_3_TEMPLATE,
    STAGE_4_TEMPLATE,
    STAGE_5_TEMPLATE
)

def build_chains(llm) -> dict:
    """Wire each PromptTemplate to the LLM and a string output parser."""
    parser = StrOutputParser()
    return {
        1: STAGE_1_TEMPLATE | llm | parser,
        2: STAGE_2_TEMPLATE | llm | parser,
        3: STAGE_3_TEMPLATE | llm | parser,
        4: STAGE_4_TEMPLATE | llm | parser,
        5: STAGE_5_TEMPLATE | llm | parser,
    }

def run_stage_with_hitl(stage_number: int, stage_name: str, chain: RunnableSequence, invoke_kwargs: dict) -> str:
    """
    Runs a single pipeline stage inside a Human-in-the-Loop approval loop.
    """
    human_feedback = ""  # empty on the first attempt

    while True:
        print(f"\n{'=' * 80}")
        print(f" STAGE {stage_number}: {stage_name}")
        print(f"{'=' * 80}")

        # Merge the caller-provided variables with the current feedback string.
        kwargs = dict(invoke_kwargs)
        kwargs["human_feedback"] = human_feedback if human_feedback else "(none)"

        # Invoke the LLM chain for this stage
        output = chain.invoke(kwargs)

        print(f"\n--- Generated Output (Stage {stage_number}) ---\n")
        print(output)
        print(f"\n{'-' * 80}")

        # HITL gate
        user_response = input(
            "\nApprove output? (Type 'y' to proceed, or type your feedback to regenerate): "
        ).strip()

        if user_response.lower() == "y":
            print(f"[Stage {stage_number} APPROVED] Proceeding to next stage...\n")
            return output
        else:
            human_feedback = user_response
            print(f"[Stage {stage_number} REJECTED] Regenerating with your feedback...")

def save_state(state: dict, path: str = "pipeline_state.json") -> None:
    """Saves the current pipeline state to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)