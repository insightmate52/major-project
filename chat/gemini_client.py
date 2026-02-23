import os
import time
from dotenv import load_dotenv
from google import genai

# Load environment variables first
load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Global token counter
TOTAL_TOKENS_USED = 0


def ask_gemini(prompt: str) -> str:
    global TOTAL_TOKENS_USED

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config={
                "max_output_tokens": 400,
                "temperature": 0.4,
            }
        )

        time.sleep(1)

        # Token usage (if available)
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