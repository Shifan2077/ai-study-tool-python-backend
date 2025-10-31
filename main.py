from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
from typing import List
from important_questions_analysis import ImportantQuestionsAnalyzer  # your existing script
import uuid

app = FastAPI()

# Allow CORS (for Angular)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_pdfs(files: List[UploadFile] = File(...)):
    pdf_paths = []
    for file in files:
        file_id = str(uuid.uuid4()) + "_" + file.filename
        file_path = os.path.join(UPLOAD_DIR, file_id)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        pdf_paths.append(file_path)

    analyzer = ImportantQuestionsAnalyzer()

    for pdf_path in pdf_paths:
        analyzer.add_questions_from_pdf(pdf_path)

    important_questions = analyzer.get_important_questions(similarity_threshold=0.65, top_n=20)

    result = {
        "files": [os.path.basename(p) for p in pdf_paths],
        "questions": [q["question"] for q in important_questions],
    }

    return result
