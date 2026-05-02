from google import genai
import os

from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def explain_prediction(user_data: dict, prediction: int, probability: float):
    
    prompt = f"""
You are a medical assistant AI.

Explain the diabetes prediction in simple terms for a patient.

Rules:
- Do NOT give medical diagnosis
- Be simple and short
- Focus on risk factors
- Be calm and non-alarming
- The provided data is your main source of response

User data:
{user_data}

Prediction:
{prediction}

Probability:
{probability}

Give a short explanation and advice.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        if not response or not response.text:
            return "Sorry, I couldn't generate an explanation."

        return response.text.strip()

    except Exception as e:
        print(f"[Gemini Error]: {e}")

        return "Sorry, the explanation service is currently unavailable. Please try again later."
