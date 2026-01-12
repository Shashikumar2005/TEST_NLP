import re

def clean_text(text):
    if not text:
        return ""

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    lower = text.lower()

    for stop in ["references", "bibliography", "acknowledgements"]:
        if stop in lower:
            idx = lower.rfind(stop)
            text = text[:idx]
            break

    return text.strip()
