from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv

load_dotenv()

llm = OpenAI()

code_prompt = PromptTemplate(
   template="Write a very short {language} function that will {task}",
   input_variables=["language", "task"]
)

code_chain = LLMChain(
   llm=llm,
   prompt=code_prompt
)

result = code_chain.invoke({
   "language": "python",
   "task": "return a list of numbers"
})

print(result["text"])

