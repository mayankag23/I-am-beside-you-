import os
from typing import Optional, List

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def _use_openai() -> bool:
    return bool(OPENAI_API_KEY)


def answer_with_openai(system: str, prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=800)
    return resp.choices[0].message.content.strip()


def hf_text2text(prompt: str, model_name: str = "google/flan-t5-small") -> str:
    from transformers import pipeline

    pipe = pipeline("text2text-generation", model=model_name, device=-1)
    out = pipe(prompt, max_length=512, truncation=True)
    return out[0]["generated_text"].strip()


def generate_notes(context: str, n_sentences: int = 12) -> str:
    """Generate short notes from context using OpenAI if available, otherwise HF fallback."""
    prompt = (
        "Generate concise lecture notes (bullet points). "
        f"Keep it short (about {n_sentences} bullets) and focused on key concepts.\n\nContext:\n{context}"
    )
    system = "You are an assistant that produces concise study notes from lecture material."
    if _use_openai():
        return answer_with_openai(system, prompt)
    return hf_text2text(prompt)


def generate_questions(context: str, n: int = 10) -> str:
    prompt = (
        "Generate study questions (mixture of short answer and conceptual) from the following lecture material. "
        f"Number them 1..{n}.\n\nContext:\n{context}"
    )
    system = "You are an assistant that generates study questions from lecture material."
    if _use_openai():
        return answer_with_openai(system, prompt)
    return hf_text2text(prompt)


def answer_question_with_context(question: str, context_chunks: List[str]) -> str:
    ctx = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "Use the context from the document to answer the question. If the answer is not in the context, say you don't know. "
        f"Context:\n{ctx}\n\nQuestion: {question}"
    )
    system = "You are a helpful assistant that answers based only on provided context. If asked beyond the context, say you don't know. Provide short, precise answers and cite the context when helpful."
    if _use_openai():
        return answer_with_openai(system, prompt)
    return hf_text2text(prompt)
