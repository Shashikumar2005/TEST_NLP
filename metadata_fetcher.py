import requests
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"

def fetch_papers(query, max_results=5):
    try:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results
        }

        response = requests.get(ARXIV_API, params=params, timeout=20)
        response.raise_for_status()

        root = ET.fromstring(response.text)

        papers = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip()
            abstract = entry.find("atom:summary", ns).text.strip()
            paper_id = entry.find("atom:id", ns).text.strip()

            pdf_url = paper_id.replace("abs", "pdf") + ".pdf"

            papers.append({
                "title": title,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "source": "arxiv"
            })

        return papers

    except Exception as e:
        print("WARNING: arXiv fetch failed:", e)
        return []
