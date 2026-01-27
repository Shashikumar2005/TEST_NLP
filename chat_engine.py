from rag_engine import search_chunks
from llm_client import call_llm


def detect_intent(q):
    q = q.lower()
    if "compare" in q or "difference" in q:
        return "comparison"
    if "summarize" in q or "summary" in q:
        return "summary"
    return "qa"


def answer_question(question, selected_titles=None):
    intent = detect_intent(question)

    chunks = search_chunks(question, top_k=12, selected_titles=selected_titles)

    if not chunks:
        return "No relevant content found in the papers."

    context = ""
    used_titles = []

    for i, c in enumerate(chunks):
        context += f"\n[Source {i+1} | {c['title']}]\n{c['text']}\n"
        if c["title"] not in used_titles:
            used_titles.append(c["title"])

    if intent == "summary":
        prompt = f"""
You are an academic research assistant.
Summarize the following papers using ONLY the provided context.
If something is not present, say "Not found in the papers".

Context:
{context}
"""

    elif intent == "comparison":
        prompt = f"""
You are an academic research assistant.
Compare the following papers based ONLY on the provided context.
If something is not present, say "Not found in the papers".

Context:
{context}
"""

    else:
        prompt = f"""
You are an academic research assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say "Not found in the papers".

Question: {question}

Context:
{context}
"""

    answer = call_llm(prompt)

    answer += "\n\nSources:\n"
    for i, t in enumerate(used_titles):
        answer += f"[{i+1}] {t}\n"

    return answer
