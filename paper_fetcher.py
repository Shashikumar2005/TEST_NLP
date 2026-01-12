from metadata_fetcher import fetch_papers
from semantic_fetcher import fetch_semantic_papers


def fetch_all_papers(query, max_papers=5):
    """
    Fetch papers from multiple sources and return unified list.
    """

    papers = []

    # ---- Fetch from arXiv ----
    try:
        arxiv_papers = fetch_papers(query, max_results=max_papers)
        papers.extend(arxiv_papers)
    except Exception as e:
        print("⚠️ arXiv fetch failed:", e)

    # ---- Fetch from Semantic Scholar ----
    try:
        sem_papers = fetch_semantic_papers(query, limit=max_papers)
        papers.extend(sem_papers)
    except Exception as e:
        print("⚠️ Semantic Scholar fetch failed:", e)

    # ---- Deduplicate by title ----
    seen = set()
    unique = []

    for p in papers:
        title = p["title"].strip().lower()
        if title in seen:
            continue
        seen.add(title)
        unique.append(p)

    # ---- Limit final count ----
    return unique[:max_papers]
