import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Create model instance
model = genai.GenerativeModel("gemini-1.5-flash")

# Global token counter
TOTAL_TOKENS_USED = 0


def ask_gemini(prompt: str) -> str:
    global TOTAL_TOKENS_USED

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 400,
                "temperature": 0.4,
            }
        )

        time.sleep(1)

        # Safe usage metadata extraction
        usage = getattr(response, "usage_metadata", None)

        if usage:
            TOTAL_TOKENS_USED += usage.total_token_count
            print("Prompt Tokens:", usage.prompt_token_count)
            print("Output Tokens:", usage.candidates_token_count)
            print("Total Tokens:", usage.total_token_count)
            print("🔥 Total Tokens Used So Far:", TOTAL_TOKENS_USED)

        return response.text.strip()

    except Exception as e:
        return f"Gemini Error: {str(e)}"