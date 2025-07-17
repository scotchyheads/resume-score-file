import fitz  
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Step 1: Extract text from all job descriptions
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Load job descriptions
job_folder = Path("job description")
job_files = sorted(job_folder.glob("*.pdf"))
job_descriptions = [extract_text_from_pdf(jd) for jd in job_files]
job_names = [jd.name for jd in job_files]

# Load resumes
resume_folder = Path("pdf_resumes")
resume_files = sorted(resume_folder.glob("*.pdf"))
resumes = [extract_text_from_pdf(resume) for resume in resume_files]
resume_names = [r.name for r in resume_files]

# Step 2: Score resumes for each job description
scores = []
for jd_text in job_descriptions:
    documents = [jd_text] + resumes
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    scores.append([round(score * 100, 2) for score in similarity])

# Step 3: Format results
df = pd.DataFrame(scores, index=[f"{i+1}. {name}" for i, name in enumerate(job_names)],
                  columns=resume_names)

print("\n📊 Resume Matching Scores (%):\n")
print(df)