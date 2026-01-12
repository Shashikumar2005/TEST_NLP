import os
from dotenv import load_dotenv
from groq import Groq

# Load .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found in .env file")

client = Groq(api_key=GROQ_API_KEY)

# Current fast Groq model
MODEL = "llama-3.1-8b-instant"

def call_llm(prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant. Answer ONLY from the given context. If the answer is not in the context, say so."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=300
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"[LLM ERROR]: {str(e)}"
