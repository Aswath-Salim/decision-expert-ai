import os
import warnings
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

warnings.filterwarnings("ignore")
load_dotenv()

# --------------------------------------------------
# LLM (Gemini)
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# --------------------------------------------------
# Tool (Tavily)
# --------------------------------------------------
tavily = TavilySearch(k=4)

# --------------------------------------------------
# Prompt template
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict senior decision-making expert."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# --------------------------------------------------
# Memory store (session-based)
# --------------------------------------------------
_store = {}

def _get_session_history(session_id: str):
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

# --------------------------------------------------
# Runnable agent with memory
# --------------------------------------------------
agent = RunnableWithMessageHistory(
    prompt | llm,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ==================================================
# PUBLIC FUNCTIONS (USED BY FLASK)
# ==================================================

def generate_questions(problem: str):
    """
    Generate EXACTLY 6 follow-up questions.
    """
    q_prompt = f"""
Decision:
"{problem}"

Generate EXACTLY 6 follow-up questions.

Rules:
- Specific to this decision
- Cover money, risks, feasibility, alternatives, timeline, backup
- Output ONLY numbered questions (1–6)
"""

    result = agent.invoke(
        {"input": q_prompt},
        config={"configurable": {"session_id": "question_gen"}}
    )

    lines = result.content.split("\n")
    questions = []

    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            questions.append(line.split(".", 1)[-1].strip())

    while len(questions) < 6:
        questions.append("Please provide more details related to this decision.")

    return questions[:6]


def final_analysis(problem: str, answers: list):
    """
    Final strict decision evaluation.
    """
    answers_text = "\n".join(
        f"{i+1}. {a}" for i, a in enumerate(answers)
    )

    final_prompt = f"""
Decision:
"{problem}"

User Answers:
{answers_text}

Evaluate this decision STRICTLY.

Step 1: Score each (0–10)
- Financial feasibility
- Risk level (10 = very risky)
- Practicality
- Backup strength
- Timeline realism

Step 2: Rules
- Financial < 4 → NOT GOOD
- Risk > 7 and weak backup → NOT GOOD
- 2+ scores below 4 → NOT GOOD
- All ≥ 6 → GOOD
- Else → CONDITIONALLY GOOD

Respond ONLY in this format:

Scores:
- Financial feasibility: X/10
- Risk level: X/10
- Practicality: X/10
- Backup strength: X/10
- Timeline realism: X/10

Decision Verdict:
Good / Not Good / Conditionally Good

Reasoning:
- Bullet points

Risks:
- Bullet points

Expert Suggestions:
- Bullet points
"""

    result = agent.invoke(
        {"input": final_prompt},
        config={"configurable": {"session_id": "final_analysis"}}
    )

    return result.content
