from langchain_openai import OpenAI
from dotenv import load_dotenv

# loads OPENAI_API_KEY from .env file
load_dotenv()

llm = OpenAI()

result = llm.invoke('Write a haiku about spring.')

print(result)
