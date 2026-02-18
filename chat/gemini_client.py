import os
import google.generativeai as genai
from dotenv import load_dotenv
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def ask_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

load_dotenv()

# 🔹 Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 🔹 Global Token Counter
TOTAL_TOKENS_USED = 0


def ask_gemini(prompt: str) -> str:
    global TOTAL_TOKENS_USED

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "max_output_tokens": 400,
            "temperature": 0.4,
        }
    )

    import time
    time.sleep(1)

    # 🔹 Extract usage metadata safely
    usage = getattr(response, "usage_metadata", None)

    if usage:
        TOTAL_TOKENS_USED += usage.total_token_count
        print("Prompt Tokens:", usage.prompt_token_count)
        print("Output Tokens:", usage.candidates_token_count)
        print("Total Tokens:", usage.total_token_count)
        print("🔥 Total Tokens Used So Far:", TOTAL_TOKENS_USED)

    return response.text.strip()