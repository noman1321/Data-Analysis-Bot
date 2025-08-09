import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def get_gemini_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
