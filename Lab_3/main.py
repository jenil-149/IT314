import os
from datetime import datetime
from dotenv import load_dotenv

from llm_config import get_llm
from utils import build_chains, run_stage_with_hitl, save_state


load_dotenv()

def main():
    print("=" * 80)
    print(" AUTOMATED REQUIREMENTS ELICITATION PIPELINE (LangChain + HITL)")
    print("=" * 80)

    # Read the case study from the external text file
    case_study_path = "case_study.txt"
    if os.path.isfile(case_study_path):
        with open(case_study_path, "r", encoding="utf-8") as f:
            case_study = f.read()
        print(f"\nLoaded case study from: {case_study_path}")
    else:
        print(f"\nERROR: Could not find '{case_study_path}'. Please create it.")
        return

    # Initialize LLM and build chains
    llm = get_llm()
    chains = build_chains(llm)

    # Initialize state tracking dictionary
    state = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "case_study": case_study,
    }

    # ---------------- STAGE 1: Stakeholder Identification ----------------
    stakeholders = run_stage_with_hitl(
        stage_number=1,
        stage_name="Stakeholder Identification",
        chain=chains[1],
        invoke_kwargs={"case_study": case_study},
    )
    state["stage_1_stakeholders"] = stakeholders
    save_state(state)

    # ---------------- STAGE 2: Stakeholder Goals ----------------
    stakeholder_goals = run_stage_with_hitl(
        stage_number=2,
        stage_name="Stakeholder Goals",
        chain=chains[2],
        invoke_kwargs={"case_study": case_study, "stakeholders": stakeholders},
    )
    state["stage_2_stakeholder_goals"] = stakeholder_goals
    save_state(state)

    # ---------------- STAGE 3: Elicitation Technique Selection ----------------
    elicitation_techniques = run_stage_with_hitl(
        stage_number=3,
        stage_name="Elicitation Technique Selection",
        chain=chains[3],
        invoke_kwargs={"stakeholder_goals": stakeholder_goals},
    )
    state["stage_3_elicitation_techniques"] = elicitation_techniques
    save_state(state)

    # ---------------- STAGE 4: Elicitation Execution ----------------
    elicitation_execution = run_stage_with_hitl(
        stage_number=4,
        stage_name="Elicitation Execution",
        chain=chains[4],
        invoke_kwargs={"elicitation_techniques": elicitation_techniques},
    )
    state["stage_4_elicitation_execution"] = elicitation_execution
    save_state(state)

    # ---------------- STAGE 5: FR/NFR Generation ----------------
    fr_nfr_output = run_stage_with_hitl(
        stage_number=5,
        stage_name="FR/NFR Generation",
        chain=chains[5],
        invoke_kwargs={
            "case_study": case_study,
            "stakeholders": stakeholders,
            "stakeholder_goals": stakeholder_goals,
            "elicitation_execution": elicitation_execution,
        },
    )
    state["stage_5_fr_nfr"] = fr_nfr_output
    save_state(state)

    # ---------------- FINAL OUTPUT ----------------
    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETE — FINAL FR/NFR OUTPUT")
    print("=" * 80 + "\n")
    print(fr_nfr_output)

    print(f"\nFull pipeline state (all approved stage outputs) saved to: pipeline_state.json")

if __name__ == "__main__":
    main()