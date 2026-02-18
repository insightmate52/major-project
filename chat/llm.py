import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

def run_llm(user_question: str, context: dict | None = None) -> str:
    """
    AI-powered response generator using local Ollama LLM.
    Uses dataset context strictly to avoid hallucination.
    """

    if not context:
        return (
            "Please upload a dataset first so I can analyze it "
            "and answer your questions."
        )

    system_prompt = f"""
You are an AI data analysis assistant.

STRICT RULES:
- Answer ONLY using the given dataset context
- Do NOT create or assume new numbers
- Do NOT generate predictions
- If information is missing, say so clearly

DATASET CONTEXT (summary):
Rows: {context.get("rows")}
Columns: {context.get("columns")}
Insights: {context.get("text_insights")}
Forecast: {context.get("forecast_result")}

USER QUESTION:
{user_question}

Provide a clear, concise, business-friendly explanation.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": system_prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json().get("response", "").strip()

    except Exception as e:
        return f"AI service unavailable. Falling back to system response. ({e})"




