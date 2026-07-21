from importlib import import_module
import os

from dotenv import load_dotenv
from openai import OpenAI

# استيراد دالة بناء الـ Context من الموديول الخاص بك
build_context = import_module("06_retrieve_context").build_context

load_dotenv()

# سحب المفتاح من البيئة (.env) لضمان الأمان
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def build_prompt(question, context):
    return f"""You are a helpful, accurate, and detailed customer support assistant.

Use ONLY the provided context to answer the question.
Your answer must be comprehensive, clear, and structured step-by-step.
Include all relevant details, conditions, steps, and options present in the context.
Do NOT give brief or summarized answers when full details are available.

If the context does not contain enough information to answer the question, state clearly that you do not have enough information.
Prefer CURRENT sources over OUTDATED sources.
Always cite the sources you used in your answer using the format [Source X].

Question:
{question}

Context:
{context}
"""


def ask_openrouter(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def answer_question(question):
    context, sources = build_context(question)

    if not OPENROUTER_API_KEY:
        return "Missing OPENROUTER_API_KEY in environment variables.", sources

    prompt = build_prompt(question, context)
    return ask_openrouter(prompt), sources

    return ask_openrouter(prompt), sources
