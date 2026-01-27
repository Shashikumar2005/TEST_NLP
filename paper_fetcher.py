from metadata_fetcher import fetch_papers
from semantic_fetcher import fetch_semantic_papers


def fetch_all_papers(query, limit=5):
    all_papers = []

    try:
        all_papers.extend(fetch_papers(query, max_results=limit))
    except Exception as e:
        print("WARNING: Arxiv fetch failed:", e)

    try:
        all_papers.extend(fetch_semantic_papers(query, limit=limit))
    except Exception as e:
        print("WARNING: Semantic Scholar fetch failed:", e)

    # Remove duplicates by title
    seen = set()
    unique = []

    for p in all_papers:
        title = p.get("title", "").lower()
        if title and title not in seen:
            seen.add(title)
            unique.append(p)

    return unique[:limit]
