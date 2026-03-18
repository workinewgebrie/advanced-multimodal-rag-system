from langchain.llms import OpenAI
from generation.prompt import build_prompt

llm = OpenAI(temperature=0)

def generate_answer(context: str, question: str) -> str:
    prompt = build_prompt(context, question)
    return llm(prompt)