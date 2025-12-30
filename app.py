from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from agent import generate_questions, final_analysis

load_dotenv()

app = Flask(__name__)
app.secret_key = "smart-decision-secret"

@app.route("/")
def home():
    session.clear()
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please say something."})

    # First message = decision problem
    if "problem" not in session:
        session["problem"] = user_msg
        session["questions"] = generate_questions(user_msg)
        session["answers"] = []
        session["index"] = 0
        return jsonify({"reply": session["questions"][0]})

    # Store answer
    session["answers"].append(user_msg)
    session["index"] += 1

    # Ask next question
    if session["index"] < len(session["questions"]):
        return jsonify({"reply": session["questions"][session["index"]]})

    # Final analysis
    result = final_analysis(
        session["problem"],
        session["answers"]
    )
    session.clear()
    return jsonify({"reply": result})

if __name__ == "__main__":
    app.run(debug=True)
