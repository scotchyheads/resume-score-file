# First, I am importing all the libraries I need

from pathlib import Path  # to work with files and folders
import fitz  # this is PyMuPDF, used to read text from PDF files
import pandas as pd  # to store and show scores in table format
from sklearn.feature_extraction.text import TfidfVectorizer  # to convert text to numbers
from sklearn.metrics.pairwise import cosine_similarity  # to compare texts and get matching score

# I made a function to read text from a PDF file
def read_pdf(file_path):
    doc = fitz.open(file_path)  # open the PDF
    text = ""
    for page in doc:
        text += page.get_text()  # get text from each page and add to one string
    doc.close()
    return text  # return the full text

# Now I set the folder where my resume and job description PDFs are kept
resume_folder = Path("pdf_resumes")  # this folder should contain all the resume PDFs
jd_folder = Path("job description")  # this folder should have job descriptions

# I collect all PDF file names in lists
resume_files = list(resume_folder.glob("*.pdf"))
jd_files = list(jd_folder.glob("*.pdf"))

# Now I read the text from all the resumes and job descriptions
resume_texts = [read_pdf(file) for file in resume_files]
jd_texts = [read_pdf(file) for file in jd_files]

# Combine all texts (job descriptions + resumes) in one list
all_texts = jd_texts + resume_texts

# I use TF-IDF to turn text into numbers
# It ignores common words like "the", "is", etc.
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Now I separate the vectors: job description vectors and resume vectors
jd_vectors = tfidf_matrix[:len(jd_texts)]  # first few are JDs
resume_vectors = tfidf_matrix[len(jd_texts):]  # rest are resumes

# Now I compare each job description to each resume using cosine similarity
# It will give a score between 0 and 1 — so I multiply by 100 to get percentage
match_scores = cosine_similarity(jd_vectors, resume_vectors) * 100

# I want to show these scores nicely in a table
# Each row is a job description, each column is a resume
score_table = pd.DataFrame(
    match_scores,
    index=[f"JD_{i+1}" for i in range(len(jd_texts))],
    columns=[f"Resume_{i+1}" for i in range(len(resume_texts))]
)

# Finally, I print the scores
print("✅ Resume Matching Scores (%):\n")
print(score_table.round(2))  # show only 2 decimal digits
