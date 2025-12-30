import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch  # ✅ correct class
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Tavily Search (safe, stable)
tavily = TavilySearch(k=4)

# ----------------------------------
# Generate 6 AI-driven questions
# ----------------------------------
def generate_questions(problem: str):
    prompt = f"""
You are a senior decision-making expert.

User decision:
"{problem}"

Generate EXACTLY 6 follow-up questions to evaluate
whether this decision is good or bad.

Rules:
- Questions must be specific to the decision
- Cover money, feasibility, risks, alternatives, timeline, backup
- Output ONLY numbered questions (1–6)
"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    questions = []
    for line in text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            questions.append(line.split(".", 1)[-1].strip())

    while len(questions) < 6:
        questions.append("Please provide more details related to this decision.")

    return questions[:6]

# ----------------------------------
# Final decision analysis (with Tavily)
# ----------------------------------
def final_analysis(problem: str, answers: list):
    try:
        search_results = tavily.invoke(problem)
    except Exception as e:
        search_results = f"No external data available ({e})"

    prompt = f"""
Decision Problem:
{problem}

User Answers:
{answers}

Relevant Real-World Information:
{search_results}

You are a STRICT decision evaluator.

Step 1: Rate each factor from 0 to 10
- Financial feasibility
- Risk level (10 = very risky)
- Practicality
- Backup/fallback strength
- Timeline realism

Step 2: Apply these rules STRICTLY:
- If financial feasibility < 4 → Decision is NOT GOOD
- If risk level > 7 AND no strong backup → NOT GOOD
- If 2 or more factors score below 4 → NOT GOOD
- If all factors ≥ 6 → GOOD
- Else → CONDITIONALLY GOOD

Step 3: Respond ONLY in this format:

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

    return llm.invoke(prompt).content.strip()
