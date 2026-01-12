import requests
import tempfile
import os

def download_pdf(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(r.content)
        tmp.close()

        return tmp.name
    except Exception as e:
        print("PDF download failed:", e)
        return None
