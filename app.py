from flask import Flask, render_template, request, jsonify
from rag_engine import build_index
from chat_engine import answer_question
import pickle

app = Flask(__name__)

# Global state
INDEX_READY = False
PAPERS = []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/build", methods=["POST"])
def build():
    global INDEX_READY, PAPERS

    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query missing"}), 400

    ok = build_index(query, max_papers=5)

    if not ok:
        return jsonify({"error": "Failed to build index"}), 500

    # Load metadata to extract paper list
    try:
        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)

        # Build unique paper list
        seen = set()
        papers = []

        for m in meta:
            title = m["title"]
            pdf = m["pdf_url"]

            if title not in seen:
                seen.add(title)
                papers.append({
                    "title": title,
                    "pdf_url": pdf
                })

        PAPERS = papers
        INDEX_READY = True

        return jsonify({
            "status": "Index built successfully!",
            "papers": PAPERS
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    global INDEX_READY

    if not INDEX_READY:
        return jsonify({"error": "Index not built yet"}), 400

    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question missing"}), 400

    answer = answer_question(question)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
