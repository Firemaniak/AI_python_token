from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API-ключ не найден. Проверь файл .env")

client = genai.Client(api_key=api_key)

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Объясни простыми словами, что такое токенизация в NLP"
)

print(interaction.output_text)