from rag_engine import search_chunks
from llm_client import call_llm

# ----------------------------
# Intent Detection
# ----------------------------

def detect_intent(question: str):
    q = question.lower()

    if "compare" in q or "difference" in q or "vs" in q:
        return "compare"
    if "summarize" in q or "summary" in q or "overview" in q:
        return "summary"
    return "chat"


# ----------------------------
# Prompt Builders
# ----------------------------

def build_chat_prompt(context_blocks, question):
    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a research assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "Not found in the provided papers."

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt


def build_summary_prompt(grouped_context):
    text = ""

    for i, (title, chunks) in enumerate(grouped_context.items(), 1):
        text += f"\nPaper {i}: {title}\n"
        text += "\n".join(chunks[:5])  # limit per paper

    prompt = f"""
You are a research assistant.

Summarize EACH paper separately using ONLY the context below.

Context:
{text}

Write output in this format:

Paper 1:
Summary...

Paper 2:
Summary...

If something is missing, say so.

Answer:
"""
    return prompt


def build_compare_prompt(grouped_context, question):
    text = ""

    for i, (title, chunks) in enumerate(grouped_context.items(), 1):
        text += f"\nPaper {i}: {title}\n"
        text += "\n".join(chunks[:5])

    prompt = f"""
You are a research assistant.

Compare the following papers using ONLY the context below.

Context:
{text}

Question:
{question}

Write a structured comparison.

If something is missing, say so.

Answer:
"""
    return prompt


# ----------------------------
# Main Chat Function
# ----------------------------

def answer_question(question: str, top_k=6):
    """
    Main entry point for answering user questions.
    """

    # 1. Detect intent
    intent = detect_intent(question)

    # 2. Retrieve relevant chunks
    results = search_chunks(question, top_k=top_k)

    if not results:
        return "No relevant content found in papers."

    # 3. Group chunks by paper
    grouped = {}

    for r in results:
        title = r["title"]
        chunk = r["text"]

        if title not in grouped:
            grouped[title] = []

        grouped[title].append(chunk)

    # 4. Build prompts based on intent
    if intent == "summary":
        prompt = build_summary_prompt(grouped)

    elif intent == "compare":
        prompt = build_compare_prompt(grouped, question)

    else:
        # Normal chat / QA
        context_blocks = []
        sources = []

        for i, (title, chunks) in enumerate(grouped.items(), 1):
            block = f"[{i}] {title}\n" + "\n".join(chunks[:3])
            context_blocks.append(block)
            sources.append(f"[{i}] {title}")

        prompt = build_chat_prompt(context_blocks, question)

    # 5. Call LLM
    answer = call_llm(prompt)

    # 6. Append citations
    citations = "\n\nSources:\n"
    for i, title in enumerate(grouped.keys(), 1):
        citations += f"[{i}] {title}\n"

    return answer.strip() + citations
