from flask import Flask, request, jsonify
import os
import docx
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# ---------------------- CONFIG ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------- MODELS ----------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# client = OpenAI()

# ---------------------- TEXT EXTRACTION ----------------------
def extract_text(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

# ---------------------- CHUNKING ----------------------
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---------------------- EMBEDDINGS ----------------------
def get_embeddings(chunks):
    return embedding_model.encode(chunks)

# ---------------------- SIMILARITY ----------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------------- RETRIEVAL ----------------------
def retrieve(query, chunks, embeddings, top_k=3):
    query_vec = embedding_model.encode([query])[0]

    scores = [
        (chunk, cosine_similarity(query_vec, emb))
        for chunk, emb in zip(chunks, embeddings)
    ]

    scores.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scores[:top_k]]

# ---------------------- PROMPT ----------------------
def build_prompt(context):
    return f"""
Extract the following details from the resume:

- Education
- Skills
- Experience
- Projects
- Domain

Return ONLY valid JSON format like:
{{
  "education": "...",
  "skills": "...",
  "experience": "...",
  "projects": "...",
  "domain": "..."
}}

Resume Content:
{context}
"""

# ---------------------- LLM CALL ----------------------
def call_llm(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()
        return result.get("response", "")

    except Exception as e:
        logger.error(f"TinyLlama Error: {e}")
        return "{}"

# ---------------------- MAIN PIPELINE ----------------------
def process_resume_llm(file_path):
    logger.info(f"Processing {file_path}")

    text = extract_text(file_path)
    if not text:
        return {}

    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    queries = {
        "education": "education degrees qualifications",
        "skills": "technical skills programming languages",
        "experience": "work experience companies roles",
        "projects": "projects development work",
        "domain": "industry domain field"
    }

    context_parts = []

    for query in queries.values():
        top_chunks = retrieve(query, chunks, embeddings)
        context_parts.extend(top_chunks)

    context = " ".join(context_parts)

    prompt = build_prompt(context)
    response = call_llm(prompt)

    return response

# ---------------------- FILE HANDLER ----------------------
def handle_file(file_path):
    file_name = os.path.basename(file_path)

    print(f"Starting with {file_name}")

    result = process_resume_llm(file_path)

    print(f"Ending with {file_name}")

    return {
        "file_name": file_name,
        "extracted_fields": result
    }

# ---------------------- API ----------------------
@app.route("/extract_section", methods=["POST"])
def extract_sections():
    try:
        data = request.get_json()
        folder_path = data.get("folder_path")

        if not folder_path:
            return jsonify({"error": "folder_path is required"}), 400

        if not os.path.exists(folder_path):
            return jsonify({"error": "Folder not found"}), 400

        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".docx")
        ]

        if not files:
            return jsonify({"error": "No DOCX files found"}), 400

        results = []

        for file in files:
            res = handle_file(file)
            results.append(res)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        logger.info("Request completed")

# ---------------------- RUN ----------------------
if __name__ == "__main__":
    app.run(debug=True)