import os
import ast
import pandas as pd
import numpy as np
from io import BytesIO
from flask import Blueprint, request, jsonify, session
from supabase import create_client
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

from .gemini_client import ask_gemini
from .prompts import build_planning_prompt


# ===========================
# 🔧 ENV SETUP
# ===========================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "uploads")

supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ===========================
# 💬 BLUEPRINT
# ===========================

chat_bp = Blueprint("chat", __name__)


# ===========================
# 💬 CHAT API
# ===========================

@chat_bp.route("/ask", methods=["POST"])
def chat_api():

    # 🔹 Check dataset
    if "dataset_key" not in session:
        return jsonify({"error": "No dataset uploaded"}), 400

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Message is required"}), 400

    # ✅ ADD: Extract question + context
    user_question = data["message"]
    context = data.get("context")
    lower_q = user_question.lower()

    # ===========================
    # 📂 LOAD DATASET
    # ===========================

    try:
        file_data = supabase_admin.storage.from_(SUPABASE_BUCKET).download(
            session["dataset_key"]
        )
        df = pd.read_csv(BytesIO(file_data))
    except Exception as e:
        return jsonify({"error": f"Dataset loading failed: {str(e)}"}), 500

    schema_info = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

        # ✅ ADD: Graph-aware context injection
    if context == "insights_page":
            visual_summary = session.get("insight_visuals", [])
            red_flags = session.get("red_flag_visuals", [])
            visual_text = ""
            for v in visual_summary:
                if isinstance(v, dict):
                    visual_text += f"- {v.get('explanation','')} (Severity: {v.get('severity','')})\n"
            red_flag_text = ""
            for r in red_flags:
                if isinstance(r, dict):
                    red_flag_text += f"- {r.get('explanation','')} (Severity: {r.get('severity','')})\n"
            graph_context_text = f"""
    Thefollowing visual insights were generated:
    
    {visual_text}
    
    Red Flag Findings:
    
    {red_flag_text}
    
    Answer the question strictly based on these findings.
    """
    else:
            graph_context_text = ""

    # ===========================
    # 🔍 INTENT DETECTION
    # ===========================

    analysis_keywords = [
        "total", "sum", "average", "mean", "max", "min",
        "count", "how many", "forecast", "predict",
        "compare", "growth", "trend",
        "highest", "lowest", "top", "bottom",
        "percentage", "distribution"
    ]

    # ===========================
    # 🧠 EXPLANATION MODE
    # ===========================

    if not any(word in lower_q for word in analysis_keywords):

        explanation_prompt = f"""
You are a professional data analyst.

Dataset columns:
{schema_info['columns']}

The dataset contains {len(df)} rows.

{graph_context_text}

User question:
{user_question}

Provide a clear and simple explanation.
Do NOT generate Python code.
"""

        explanation = ask_gemini(explanation_prompt)

        # ✅ ADD: Store chat history properly
        if "chat_history" not in session:
            session["chat_history"] = []

        session["chat_history"].append({
            "question": user_question,
            "answer": explanation
        })
        session.modified = True

        return jsonify({"answer": explanation})

    # ===========================
    # 📊 ANALYTICAL MODE
    # ===========================

    # ✅ ADD graph context into analytical planning prompt
    planning_prompt = build_planning_prompt(
        schema_info,
        user_question + "\n" + graph_context_text
    )

    generated_code = ask_gemini(planning_prompt)

    if not generated_code:
        return jsonify({"error": "Model did not generate code"}), 500

    generated_code = generated_code.strip()

    if "```" in generated_code:
        parts = generated_code.split("```")
        if len(parts) >= 2:
            generated_code = parts[1]

    generated_code = generated_code.replace("python", "").strip()

    print("----- GENERATED RAW RESPONSE -----")
    print(generated_code)
    print("-----------------------------------")

    try:
        ast.parse(generated_code)
    except SyntaxError as e:
        print("SYNTAX ERROR:", e)
        print("BAD CODE:\n", generated_code)
        return jsonify({"error": "Model generated invalid Python code."}), 500

    try:
        allowed_globals = {
            "df": df,
            "pd": pd,
            "np": np,
            "LinearRegression": LinearRegression
        }

        local_vars = {}

        exec(generated_code, allowed_globals, local_vars)

        result = local_vars.get("result")

        if result is None:
            for value in reversed(list(local_vars.values())):
                if isinstance(value, (pd.Series, pd.DataFrame, int, float, str, dict)):
                    result = value
                    break

        if result is None:
            return jsonify({"error": "No valid analytical result generated"}), 500

    except Exception as e:
        print("EXEC ERROR:", e)
        print("CODE WAS:\n", generated_code)
        return jsonify({
            "error": f"Execution failed: {str(e)}"
        }), 500

    # ===========================
    # 🧾 RESPONSE HANDLING
    # ===========================

    if isinstance(result, (int, float, str)):
        final_answer = f"The result is {result}."

        # ✅ ADD: Save history
        if "chat_history" not in session:
            session["chat_history"] = []

        session["chat_history"].append({
            "question": user_question,
            "answer": final_answer
        })
        session.modified = True

        return jsonify({"answer": final_answer})

    if isinstance(result, pd.Series):
        result = result.to_string()

    if isinstance(result, pd.DataFrame):
        result = result.head(20).to_string(index=False)

    explanation_prompt = f"""
The dataset analysis produced the following result:

{result}

Provide a concise, professional explanation in 2–3 lines.
Do not restate the question.
"""

    explanation = ask_gemini(explanation_prompt)

    # ✅ ADD: Save analytical explanation
    if "chat_history" not in session:
        session["chat_history"] = []

    session["chat_history"].append({
        "question": user_question,
        "answer": explanation
    })
    session.modified = True

    return jsonify({"answer": explanation})