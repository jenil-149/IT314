from langchain_core.prompts import PromptTemplate

STAGE_1_TEMPLATE = PromptTemplate(
    input_variables=["case_study", "human_feedback"],
    template="""You are an expert Business Analyst performing requirements elicitation.

    TASK: Read the system case study below and identify ALL relevant stakeholders
    (primary, secondary, and any indirect/oversight stakeholders). For each
    stakeholder, give a one-line justification for why they are a stakeholder.

    Format the output as a numbered list in plain text:
    1. <Stakeholder Name> - <one-line justification>

    CASE STUDY:
    {case_study}

    REVIEWER FEEDBACK ON A PREVIOUS ATTEMPT (empty if this is the first attempt,
    otherwise you MUST incorporate this feedback into your revised answer):
    {human_feedback}

    Produce ONLY the numbered stakeholder list.""",
)

STAGE_2_TEMPLATE = PromptTemplate(
    input_variables=["case_study", "stakeholders", "human_feedback"],
    template="""You are an expert Business Analyst performing requirements elicitation.

    TASK: For EACH stakeholder listed below (carried forward from Stage 1), identify
    their specific goals, needs, and pain points with respect to the system
    described in the case study.

    CASE STUDY:
    {case_study}

    APPROVED STAKEHOLDERS (from Stage 1):
    {stakeholders}

    REVIEWER FEEDBACK ON A PREVIOUS ATTEMPT (empty if this is the first attempt,
    otherwise you MUST incorporate this feedback into your revised answer):
    {human_feedback}

    Format the output in plain text as:
    ### <Stakeholder Name>
    - Goal/Need: ...
    - Goal/Need: ...
    - Pain Point: ...

    Produce ONLY this structured goals list, covering every stakeholder.""",
)

STAGE_3_TEMPLATE = PromptTemplate(
    input_variables=["stakeholder_goals", "human_feedback"],
    template="""You are an expert Requirements Engineer.

    TASK: Based on the stakeholders and their goals/needs below, recommend the most
    appropriate elicitation technique(s) for EACH stakeholder (e.g., interview,
    questionnaire/survey, workshop, brainstorming, observation/contextual inquiry,
    document analysis, prototyping). Justify each choice in one sentence, tying it
    to that stakeholder's role, availability, and the nature of the information
    needed.

    APPROVED STAKEHOLDER GOALS (from Stage 2):
    {stakeholder_goals}

    REVIEWER FEEDBACK ON A PREVIOUS ATTEMPT (empty if this is the first attempt,
    otherwise you MUST incorporate this feedback into your revised answer):
    {human_feedback}

    Format the output in plain text: as:
    ### <Stakeholder Name>
    - Technique(s): ...
    - Justification: ...

    Produce ONLY this structured technique-selection list.""",
)

STAGE_4_TEMPLATE = PromptTemplate(
    input_variables=["elicitation_techniques", "human_feedback"],
    template="""You are an expert Requirements Engineer executing elicitation activities.

    TASK: For EACH stakeholder and their chosen elicitation technique(s) below,
    DRAFT the actual execution artifact:
    - If "Interview" was chosen: list who specifically should be interviewed and
    write 5-8 concrete interview questions.
    - If "Questionnaire/Survey" was chosen: write 6-10 concrete survey questions
    (mix of multiple-choice, Likert-scale, and open-ended).
    - If "Workshop/Brainstorming" was chosen: list an agenda with 4-6 discussion
    topics/prompts.
    - If "Observation" was chosen: list 4-6 specific things to observe and record.

    APPROVED ELICITATION TECHNIQUE SELECTION (from Stage 3):
    {elicitation_techniques}

    REVIEWER FEEDBACK ON A PREVIOUS ATTEMPT (empty if this is the first attempt,
    otherwise you MUST incorporate this feedback into your revised answer):
    {human_feedback}

    Format the output in plain text: as:
    ### <Stakeholder Name> — <Technique>
    1. <artifact item 1>
    2. <artifact item 2>
    ...

    Produce ONLY this structured execution artifact list.""",
)

STAGE_5_TEMPLATE = PromptTemplate(
    input_variables=[
        "case_study",
        "stakeholders",
        "stakeholder_goals",
        "elicitation_execution",
        "human_feedback",
    ],
    template="""You are an expert Requirements Engineer producing a final Software
    Requirements Specification (SRS) fragment.

    TASK: Using the case study, the approved stakeholders, their goals, and the
    elicitation artifacts gathered so far, generate a structured, numbered list of
    Functional Requirements (FRs) and Non-Functional Requirements (NFRs) for the
    system.

    Each FR must follow the format:
    FR-<id>: The system shall <capability>. (Source: <stakeholder(s)>)

    Each NFR must follow the format:
    NFR-<id>: The system shall <quality attribute requirement>. (Category:
    <Performance/Security/Usability/Reliability/Scalability/etc.>)

    Cover at least: order placement/payment, inventory/demand forecasting,
    feedback/ratings, administration oversight/compliance visibility, and hostel
    mess-plan integration. Aim for 12-18 FRs and 8-12 NFRs.

    CASE STUDY:
    {case_study}

    APPROVED STAKEHOLDERS:
    {stakeholders}

    APPROVED STAKEHOLDER GOALS:
    {stakeholder_goals}

    APPROVED ELICITATION EXECUTION ARTIFACTS:
    {elicitation_execution}

    REVIEWER FEEDBACK ON A PREVIOUS ATTEMPT (empty if this is the first attempt,
    otherwise you MUST incorporate this feedback into your revised answer):
    {human_feedback}

    Produce ONLY the two headed sections "## Functional Requirements" and
    "## Non-Functional Requirements" with the numbered lists inside them.""",
)