import os
import pandas as pd
import spacy
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
from docx import Document

# Suppress symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load models
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

model = load_model()
nlp = load_spacy()

# Helper to extract text based on file type
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        return ""

# Named entity extraction (skills and companies)
def extract_entities(text):
    doc = nlp(text)
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return orgs  # Replace with actual skill extraction if available

# Compute similarity
def compute_similarity(resume_text, job_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(resume_embedding, job_embedding).item()

# Streamlit App
st.title("🧠 Resume Matcher with Semantic Search and Entity Extraction")

job_file = st.file_uploader("Upload Job Posting (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])
if job_file:
    job_text = extract_text(job_file)
    st.subheader("Job Description Preview")
    st.text_area("Job Posting", job_text, height=200)

    resume_files = st.file_uploader("Upload Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)

    if resume_files:
        results = []
        for resume in resume_files:
            try:
                resume_text = extract_text(resume)
                score = compute_similarity(resume_text, job_text)
                orgs = extract_entities(resume_text)

                results.append({
                    "Resume": resume.name,
                    "Similarity Score": round(score, 4),
                    "Companies Mentioned": ", ".join(orgs)
                })
            except Exception as e:
                st.error(f"Error processing {resume.name}: {e}")

        if results:
            df = pd.DataFrame(results).sort_values(by="Similarity Score", ascending=False)
            st.subheader("📊 Ranked Resumes")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "resume_similarity_results.csv", "text/csv")

