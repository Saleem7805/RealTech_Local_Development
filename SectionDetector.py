from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# ---------------------- CONFIG ----------------------
# This is a dictionary of regex patterns used to identify sections in a resume.
# r for raw string and b for match full word only
SECTION_PATTERNS = {
    "education": [r"\beducation\b", r"\bqualification\b"],
    "experience": [r"\bexperience\b", r"\bwork history\b"],
    "skills": [r"\bskills\b", r"\btechnical skills\b"],
    "projects": [r"\bprojects\b"],
    "secondary_skills": [r"\badditional skills\b", r"\bother skills\b"],
    "domain": [r"\bdomain\b", r"\bindustry\b"]
}

# ---------------------- LOGGING ----------------------
# Keeping log levels to INFO 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print(logger )

# ---------------------- NLP MODEL ----------------------
#Try to load model → If not available → Tell user exactly what to do 
# en_core_web_sm means en → English
# core → general-purpose model
# web → trained on web data
# sm → small model (fast, less accurate than large ones)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

# ---------------------- FASTAPI SETUP ----------------------
# This creates your main application instance
app = FastAPI(title="Resume Section Extractor")
# This creates your main application instance
router = APIRouter()

# ---------------------- REQUEST MODEL ----------------------

# "API expects input JSON with a field called file_name (string)"
class ResumeRequest(BaseModel):
    file_name: str

# ---------------------- PDF PARSER ----------------------

# Instead of plain text, we extract blocks
def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        full_text = ""

        for page in doc:
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

            for block in blocks:
                full_text += block[4] + "\n"

        return full_text

    except Exception as e:
        logger.error(f"PDF parsing error: {e}")
        raise

# ---------------------- SECTION DETECTOR ----------------------
#  Taking input as String and output as dictionary
# sections → stores final result
#  current_section → tracks where we are
# text: str → input is a string (full resume text)

def detect_sections(text: str) -> dict:
    # Initialized dictionary , Create an empty dictionary to store results
    sections = {}
    # start with a default section called "other" , Some text appears before headings (like name, email)
    current_section = "other"
    # It creates a new key in the dictionary and assigns it an empty list
    sections[current_section] = []
#     #After line 84 it will be like
#  sections = {
#     "other": []
# }
    # Break the resume into line-by-line list
    lines = text.split("\n")
    
    for line in lines:
        # strip() → remove spaces  and lower() → convert to lowercase
        line_clean = line.strip().lower()

        matched = False
        #  This code checks the section pattern and if the pattern matches we will add to sections
        for section, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_clean):
                    current_section = section
                    sections.setdefault(section, [])
                    matched = True
                    break
            if matched:
                break
        #  Every line is added to current section
        sections.setdefault(current_section, []).append(line)
    # Converts:List of lines → single string

    return {k: " ".join(v).strip() for k, v in sections.items()}

# in one line summary of this fnction
#  Text
#  ↓
# Split into lines
#  ↓
# Check each line (regex)
#  ↓
# Update section
#  ↓
# Store lines
#  ↓
# Join into string
#  ↓
# Return dictionary

# ---------------------- NLP PROCESSING ----------------------

# NLP (spaCy) step for extracting structured data
# Input: Resume text (string) , Dictionary of extracted entities
def extract_entities(text: str):
    # This sends text into spaCy NLP pipeline
    # # What happens internally:
    # # Tokenization
    # # POS tagging
    # # Named Entity Recognition (NER)

    doc = nlp(text)

    entities = {"ORG": [], "DATE": []}
    # I worked for infosys in 2022 and in tcs in 2023 
    #     {
#   "ORG": ["Infosys", "TCS"],
#   "DATE": ["2022", "2023"]
# }

    for ent in doc.ents:
        if ent.label_ in ["ORG", "DATE"]:
            entities[ent.label_].append(ent.text)
# Final output dictionary
    return entities


def tfidf_classify(sections: dict):
    # Convert dictionary values → list
    corpus = list(sections.values())

    if not corpus:
        return []
    #  Initialize TF-IDF model
    vectorizer = TfidfVectorizer(stop_words="english")
# fit() Learn vocabulary (unique words) 
# transform()
# Convert text → numbers
    tfidf_matrix = vectorizer.fit_transform(corpus)
# Convert sparse matrix → normal array
    return tfidf_matrix.toarray()

# ---------------------- PIPELINE ----------------------
# This function takes a resume PDF file path and converts it into a structured JSON output

def process_resume(file_path: str) -> dict:
    logger.info(f"Processing file: {file_path}")

    # Step 1: Extract text
    # Opens PDF Reads all text from pages  Converts it into raw string these are calling funct
    text = extract_text_from_pdf(file_path)

    # Step 2: Detect sections
    sections = detect_sections(text)

    # Step 3: NLP enrichment (optional use)
    enriched_data = {}
    for section, content in sections.items():
        enriched_data[section] = {
            "text": content,
            "entities": extract_entities(content)
        }

    # Step 4: TF-IDF (optional use)
    tfidf_scores = tfidf_classify(sections)

    # Step 5: Final Output
    return {
        "education": sections.get("education", ""),
        "skill_set": sections.get("skills", ""),
        "experience": sections.get("experience", ""),
        "projects": sections.get("projects", ""),
        "secondary_skill": sections.get("secondary_skills", ""),
        "domain": sections.get("domain", "")
    }

# ---------------------- API ROUTE ----------------------
# This defines an API endpoint
# HTTP method = POST
# URL path = /  Acts as controller 
@router.post("/")
# Input must match ResumeRequest model and it is async call 
async def extract_sections(request: ResumeRequest):
    try:
        result = process_resume(request.file_name)
# # API request
#    ↓
# request.file_name
#    ↓
# process_resume(file_name)
#    ↓
# returns structured JSON
# # 
# # 
# # 
        return {
            "file_name": request.file_name,
            "extracted_fields": result
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------- REGISTER ROUTER ----------------------
# make all routes start with /extract_section
app.include_router(router, prefix="/extract_section")