import os
os.environ["USER_AGENT"] = "AI_python_token_homework/1.0"

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.summarize import load_summarize_chain
from dotenv import load_dotenv


#------------------------------------------------------------------------------------


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API-ключ не найден. Проверь файл .env")

url = "https://ru.wikipedia.org/wiki/Обработка_естественного_языка"
loader = WebBaseLoader(url)
docs = loader.load()

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key)
chain = load_summarize_chain(llm, chain_type="stuff")

result = chain.invoke(docs)
print(result["output_text"])