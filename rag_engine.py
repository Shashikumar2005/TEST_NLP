import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from paper_fetcher import fetch_all_papers
from pdf_processor import extract_text_from_pdf
from chunker import chunk_text

# ----------------------------
# Globals
# ----------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "vector.index"
META_PATH = "meta.pkl"

model = SentenceTransformer(MODEL_NAME)

index = None
metadata = []


# ----------------------------
# Build Index
# ----------------------------

def build_index(query, max_papers=5):
    global index, metadata

    print("Fetching papers...")
    papers = fetch_all_papers(query, max_papers)


    if not papers:
        print("❌ No papers found.")
        return False

    all_chunks = []
    metadata = []

    print("Processing PDFs...")
    for i, paper in enumerate(papers, 1):
        title = paper["title"]
        pdf_url = paper["pdf_url"]

        print(f"[{i}] Downloading: {title}")

        text = extract_text_from_pdf(pdf_url)

        if not text or len(text.strip()) < 200:
            print(f"⚠️ Skipping (no text): {title}")
            continue

        chunks = chunk_text(text)

        for c in chunks:
            all_chunks.append(c)
            metadata.append({
                "title": title,
                "text": c,
                "pdf_url": pdf_url
            })

    if not all_chunks:
        print("❌ No chunks extracted.")
        return False

    print("Embedding chunks...")
    embeddings = model.encode(all_chunks, batch_size=32, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index + metadata
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Index built successfully!")
    print("Total chunks:", len(metadata))

    return True


# ----------------------------
# Load Index
# ----------------------------

def load_index():
    global index, metadata

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return False

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    return True


# ----------------------------
# Search Chunks (FOR CHAT)
# ----------------------------

def search_chunks(query, top_k=6):
    global index, metadata

    if index is None or not metadata:
        ok = load_index()
        if not ok:
            print("❌ No index found. Please run build_index first.")
            return []

    q_emb = model.encode([query])

    distances, indices = index.search(q_emb, top_k)

    results = []

    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue

        results.append(metadata[idx])

    return results


# ----------------------------
# Backward Compatibility (if needed)
# ----------------------------

def search_index(query, top_k=6):
    return search_chunks(query, top_k)
