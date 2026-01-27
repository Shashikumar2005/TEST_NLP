import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from paper_fetcher import fetch_all_papers
from pdf_processor import extract_text_from_pdf
from chunker import chunk_text

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

INDEX = None
META = []
DIM = 384


def build_index(query, max_papers=5):
    global INDEX, META

    print("Fetching papers...")
    papers = fetch_all_papers(query, limit=max_papers)


    if not papers:
        print("No papers found.")
        return False

    all_chunks = []
    meta = []

    print("Processing PDFs...")
    for i, p in enumerate(papers):
        print(f"[{i+1}] Downloading: {p['title']}")

        text = extract_text_from_pdf(p["pdf_url"])
        if not text:
            continue

        chunks = chunk_text(text)

        for c in chunks:
            all_chunks.append(c)
            meta.append({
                "text": c,
                "title": p["title"],
                "pdf_url": p["pdf_url"]
            })

    if not all_chunks:
        print("No chunks created.")
        return False

    print("Embedding chunks...")
    embeddings = EMBEDDER.encode(all_chunks, show_progress_bar=True)

    index = faiss.IndexFlatL2(DIM)
    index.add(np.array(embeddings).astype("float32"))

    INDEX = index
    META = meta

    with open("index.faiss", "wb") as f:
        pickle.dump(INDEX, f)

    with open("meta.pkl", "wb") as f:
        pickle.dump(META, f)

    print("Index built successfully!")
    print("Total chunks:", len(META))

    return True


def load_index():
    global INDEX, META

    if os.path.exists("index.faiss") and os.path.exists("meta.pkl"):
        with open("index.faiss", "rb") as f:
            INDEX = pickle.load(f)
        with open("meta.pkl", "rb") as f:
            META = pickle.load(f)


def keyword_score(text, keywords):
    text = text.lower()
    score = 0
    for k in keywords:
        if k in text:
            score += 1
    return score


def expand_query(q):
    q = q.lower()
    expansions = [q]

    if "dataset" in q:
        expansions += ["data", "benchmark", "training set", "evaluation"]

    if "method" in q:
        expansions += ["approach", "algorithm", "architecture", "model"]

    if "result" in q:
        expansions += ["accuracy", "performance", "evaluation", "experiment"]

    return list(set(expansions))


def search_chunks(query, top_k=12, selected_titles=None):
    global INDEX, META, EMBEDDER

    if INDEX is None:
        load_index()

    if INDEX is None:
        return []

    queries = expand_query(query)

    candidate_scores = {}

    for q in queries:
        q_emb = EMBEDDER.encode([q])
        D, I = INDEX.search(np.array(q_emb).astype("float32"), 40)

        for rank, idx in enumerate(I[0]):
            item = META[idx]

            if selected_titles:
                if item["title"] not in selected_titles:
                    continue

            text = item["text"]

            key_score = keyword_score(text, q.split())
            vec_score = 1 / (1 + D[0][rank])

            score = 0.7 * vec_score + 0.3 * key_score

            if idx not in candidate_scores or candidate_scores[idx] < score:
                candidate_scores[idx] = score

    ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    used_texts = set()

    for idx, score in ranked:
        t = META[idx]["text"]

        if t[:100] in used_texts:
            continue

        used_texts.add(t[:100])
        results.append(META[idx])

        if len(results) >= top_k:
            break

    return results
