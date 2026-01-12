import fitz  # PyMuPDF
import requests
import io


def extract_text_from_pdf(pdf_url: str) -> str:
    """
    Download PDF from URL and extract text using PyMuPDF.
    """

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(pdf_url, headers=headers, timeout=20)
        response.raise_for_status()

        pdf_bytes = response.content

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()

        return text

    except Exception as e:
        print(f"⚠️ PDF extraction failed: {e}")
        return ""
