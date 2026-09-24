from google import genai
from google.genai import errors
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os
import time

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API-ключ не найден. Проверь файл .env")

client = genai.Client(api_key=api_key)


MIN_INTERVAL = 1
last_request_time = 0

def wait_for_rate_limit():
    global last_request_time
    elapsed = time.time() - last_request_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    last_request_time = time.time()



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type((errors.ServerError, errors.ClientError))
)
def ask_gemini(prompt: str):
    wait_for_rate_limit()
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )
    return interaction.output_text


if __name__ == "__main__":
    try:
        answer = ask_gemini("Объясни простыми словами, что такое эмбеддинги в NLP")
        print(answer)
    except Exception as e:
        print("Не удалось получить ответ после всех попыток:", type(e).__name__, e)