from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import spacy
from spacy.matcher import PhraseMatcher
import logging
import re
import os
import docx

# ---------------------- CONFIG ----------------------

SECTION_KEYWORDS = {
    "education": ["bachelor", "master", "phd", "university", "college", "degree"],
    "experience": ["worked", "experience", "company", "role", "engineer", "developer"],
    "skills": ["python", "java", "sql", "machine learning", "communication"],
    "projects": ["project", "developed", "built", "implementation"],
    "secondary_skills": ["leadership", "teamwork", "languages", "hobbies"],
    "domain": ["banking", "finance", "healthcare", "retail", "insurance"]
}

SKILL_DB = [
    "python", "java", "sql", "excel", "power bi", "machine learning",
    "kronos", "wim", "boomi", "sap", "oracle"
]

DOMAIN_DB = ["banking", "finance", "healthcare", "retail", "insurance"]

SOFT_SKILLS = ["leadership", "communication", "teamwork"]

# ---------------------- LOGGING ----------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- LOAD SPACY ----------------------

try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

# ---------------------- MATCHER ----------------------

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

for label, words in SECTION_KEYWORDS.items():
    patterns = [nlp.make_doc(text) for text in words]
    matcher.add(label, patterns)

# ---------------------- REFERENCE TEXT ----------------------

REFERENCE_TEXT = {
    "education": nlp("degree university academic qualification"),
    "experience": nlp("work job role company experience"),
    "skills": nlp("technical skills programming languages"),
    "projects": nlp("projects developed system application"),
    "secondary_skills": nlp("soft skills leadership communication"),
    "domain": nlp("industry domain banking finance healthcare")
}

# ---------------------- FLASK APP ----------------------

app = Flask(__name__)

# ---------------------- FILE READER ----------------------

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)
        return "".join([page.get_text() for page in doc])

    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ---------------------- CHUNKING ----------------------

def get_chunks(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

# ---------------------- CLASSIFIER ----------------------

def classify_chunk(chunk):
    doc = nlp(chunk)

    matches = matcher(doc)
    if matches:
        label_counts = {}
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id]
            label_counts[label] = label_counts.get(label, 0) + 1
        return max(label_counts, key=label_counts.get)

    if re.search(r"\b\d{4}\b", chunk) and "company" in chunk.lower():
        return "experience"

    best_label = "other"
    max_score = 0

    for label, ref_doc in REFERENCE_TEXT.items():
        score = doc.similarity(ref_doc)
        if score > max_score:
            max_score = score
            best_label = label

    return best_label

# ---------------------- SECTION DETECTION ----------------------

def detect_sections_nlp(text):
    chunks = get_chunks(text)

    sections = {
        "education": [],
        "experience": [],
        "skills": [],
        "projects": [],
        "secondary_skills": [],
        "domain": [],
        "other": []
    }

    for chunk in chunks:
        label = classify_chunk(chunk)
        sections[label].append(chunk)

    return sections

# ---------------------- EXTRACTION FUNCTIONS ----------------------

def extract_skills(text):
    text_lower = text.lower()
    return list(set([skill for skill in SKILL_DB if skill in text_lower]))

def extract_domain(text):
    text_lower = text.lower()
    return list(set([d for d in DOMAIN_DB if d in text_lower]))

def extract_secondary_skills(text):
    text_lower = text.lower()
    return list(set([s for s in SOFT_SKILLS if s in text_lower]))

def extract_education(text):
    doc = nlp(text)
    education = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            if any(word in ent.text.lower() for word in ["university", "college", "institute"]):
                education.append(ent.text)

    degrees = re.findall(r"(b\.?tech|m\.?tech|mba|bachelor|master|phd)", text.lower())

    return list(set(education + degrees))

def extract_experience(text):
    doc = nlp(text)
    companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    roles = re.findall(r"(engineer|developer|analyst|consultant|manager)", text.lower())

    return list(set(companies + roles))

def extract_projects(text):
    projects = []

    for line in text.split("\n"):
        if "project" in line.lower() or "developed" in line.lower():
            projects.append(line.strip())

    return projects[:5]

# ---------------------- MAIN PIPELINE ----------------------

def process_resume(file_path: str) -> dict:
    logger.info(f"Processing file: {file_path}")

    text = extract_text(file_path)
    sections = detect_sections_nlp(text)

    return {
        "education": extract_education(" ".join(sections["education"])),
        "skill_set": extract_skills(" ".join(sections["skills"])),
        "experience": extract_experience(" ".join(sections["experience"])),
        "projects": extract_projects(" ".join(sections["projects"])),
        "secondary_skill": extract_secondary_skills(" ".join(sections["secondary_skills"])),
        "domain": extract_domain(" ".join(sections["domain"]))
    }

# ---------------------- API ----------------------

@app.route("/extract_section", methods=["POST"])
def extract_sections():
    try:
        data = request.get_json()
        file_name = data.get("file_name")

        if not file_name:
            return jsonify({"error": "file_name is required"}), 400

        result = process_resume(file_name)

        return jsonify({
            "file_name": file_name,
            "extracted_fields": result
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------- RUN ----------------------

if __name__ == "__main__":
    app.run(debug=True)