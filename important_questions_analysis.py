import re
import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#pytessereact ext for host
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ============= QUESTION EXTRACTION (from v14) =============

def extract_question_sentences(text, debug=False):
    """Extract actual question sentences from Mumbai University question paper text."""
    questions = []
    lines = text.split('\n')

    skip_patterns = [
        r'paper\s*/\s*subject code',
        r'qp code:',
        r'time:\s*\d+',
        r'marks:\s*\d+',
        r'page \d+ of \d+',
        r'semester \(sem',
        r'note:\s*\d+\.',
        r'answer any (three|four|five)',
        r'question \d+ is compulsory',
        r'assume\s+suitable\s+data.*state\s+it\s+clearly'

    ]

    question_indicators = [
        'what', 'why', 'how', 'explain', 'describe', 'compare',
        'list', 'define', 'draw', 'state', 'give', 'discuss',
        'write', 'differentiate', 'calculate', 'derive', 'prove',
        'analyze', 'evaluate', 'illustrate', 'justify', 'demonstrate',
        'show', 'find', 'determine', 'compute', 'solve', 'construct',
        'identify', 'outline', 'summarize', 'classify', 'distinguish',
        'elaborate', 'comment', 'mention', 'develop', 'implement'
    ]

    subpart_pattern = r'^[a-z]\)\s+'
    question_markers = r'^(Q\d+\s*[a-z]?\)|Q\d+)'
    marks_pattern = r'[\[\(]?\d+[\)\]]?\s*$'

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or len(line) < 5:
            i += 1
            continue

        is_metadata = any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns)

        if is_metadata:
            i += 1
            continue

        if re.match(r'^[A-Z0-9\s\-\/\(\)]+$', line) and len(line) < 35 and not re.search(r'[a-z]', line):
            i += 1
            continue

        cleaned_line = re.sub(question_markers, '', line, flags=re.IGNORECASE).strip()
        cleaned_line = re.sub(marks_pattern, '', cleaned_line).strip()

        has_q_marker = bool(re.match(r'^Q\d+', line, re.IGNORECASE))
        has_subpart_marker = bool(re.match(subpart_pattern, line))
        has_question_word = any(word in cleaned_line.lower() for word in question_indicators)

        is_question = (has_q_marker or has_subpart_marker or has_question_word) and len(cleaned_line) > 5

        if is_question and cleaned_line.endswith(':'):
            main_question = cleaned_line
            sub_parts = []

            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()

                if re.match(subpart_pattern, next_line):
                    sub_text = re.sub(subpart_pattern, '', next_line).strip()
                    sub_text = re.sub(marks_pattern, '', sub_text).strip()
                    if sub_text and len(sub_text) > 5:
                        sub_parts.append(sub_text)
                    j += 1
                elif not next_line or len(next_line) < 3:
                    j += 1
                else:
                    break

            if sub_parts:
                full_question = main_question + " " + "; ".join(sub_parts)
                if full_question not in questions:
                    questions.append(full_question)
                i = j
            else:
                i += 1

        elif is_question and len(cleaned_line) < 700:
            cleaned_line = re.sub(r'^[a-z]\)\s*', '', cleaned_line, flags=re.IGNORECASE).strip()
            cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', cleaned_line).strip()

            if cleaned_line and len(cleaned_line) > 5 and cleaned_line not in questions:
                questions.append(cleaned_line)
            i += 1
        else:
            i += 1

    return questions


def extract_text_from_pdf(pdf_path, dpi=200):
    """Extract text from PDF using pytesseract OCR."""
    print(f"  Converting to images...")
    images = convert_from_path(pdf_path, dpi=dpi)

    all_text = ""
    for i, image in enumerate(images, 1):
        print(f"  OCR on page {i}/{len(images)}...")
        custom_config = r'--oem 3 --psm 6'
        page_text = pytesseract.image_to_string(image, lang='eng', config=custom_config)
        all_text += page_text + "\n\n"

    return all_text


def extract_questions_from_pdf(pdf_path):
    """Extract questions from a single PDF."""
    text = extract_text_from_pdf(pdf_path, dpi=300)
    questions = extract_question_sentences(text, debug=False)
    return questions


# ============= IMPORTANT QUESTIONS ANALYSIS =============

class ImportantQuestionsAnalyzer:
    def __init__(self):
        self.all_questions = []
        self.question_sources = []  # Track which PDF each question came from
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Clean and preprocess text for better analysis."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def add_questions_from_pdf(self, pdf_path):
        """Extract and add questions from a PDF file."""
        print(f"\nProcessing: {os.path.basename(pdf_path)}")
        questions = extract_questions_from_pdf(pdf_path)
        print(f"  Extracted {len(questions)} questions")

        for q in questions:
            self.all_questions.append(q)
            self.question_sources.append(os.path.basename(pdf_path))

        return len(questions)

    def process_folder(self, folder_path):
        """Process all PDF files in a folder."""
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return

        print(f"\nFound {len(pdf_files)} PDF files")
        print("="*60)

        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            self.add_questions_from_pdf(pdf_path)

        print("\n" + "="*60)
        print(f"Total questions extracted: {len(self.all_questions)}")

    def find_similar_questions(self, threshold=0.7):
        """Group similar questions using cosine similarity."""
        if len(self.all_questions) < 2:
            return []

        print("\nFinding similar questions...")

        # Preprocess questions
        preprocessed = [self.preprocess_text(q) for q in self.all_questions]

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed)

        # Calculate similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Group similar questions
        groups = []
        processed = set()

        for i in range(len(self.all_questions)):
            if i in processed:
                continue

            similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
            if len(similar_indices) > 1:
                group = {
                    'representative': self.all_questions[i],
                    'similar_questions': [self.all_questions[j] for j in similar_indices if j != i],
                    'count': len(similar_indices),
                    'sources': [self.question_sources[j] for j in similar_indices]
                }
                groups.append(group)
                processed.update(similar_indices)

        # Sort by frequency
        groups.sort(key=lambda x: x['count'], reverse=True)
        return groups

    def extract_key_topics(self, top_n=20):
        """Extract key topics/terms using TF-IDF."""
        print("\nExtracting key topics...")

        preprocessed = [self.preprocess_text(q) for q in self.all_questions]

        vectorizer = TfidfVectorizer(max_features=top_n, ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform(preprocessed)

        feature_names = vectorizer.get_feature_names_out()

        # Calculate average TF-IDF score for each term
        avg_scores = tfidf_matrix.mean(axis=0).A1
        top_indices = avg_scores.argsort()[-top_n:][::-1]

        topics = [(feature_names[i], avg_scores[i]) for i in top_indices]
        return topics

    def get_important_questions(self, similarity_threshold=0.7, top_n=20):
        """Get the most important questions based on frequency and similarity."""
        similar_groups = self.find_similar_questions(threshold=similarity_threshold)

        # Get top N most frequent question patterns
        important_questions = []
        for group in similar_groups[:top_n]:
            important_questions.append({
                'question': group['representative'],
                'frequency': group['count'],
                'appears_in': list(set(group['sources']))
            })

        return important_questions

    def generate_report(self, output_file='important_questions_report.txt', similarity_threshold=0.7):
        """Generate a comprehensive report."""
        print("\n" + "="*60)
        print("GENERATING REPORT")
        print("="*60)

        report = []
        report.append("="*60)
        report.append("IMPORTANT QUESTIONS ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"\nTotal Questions Analyzed: {len(self.all_questions)}")
        report.append(f"From {len(set(self.question_sources))} question papers")
        report.append("\n" + "="*60)

        # Most Important Questions
        report.append("\n1. MOST FREQUENTLY ASKED QUESTIONS")
        report.append("="*60)
        important = self.get_important_questions(similarity_threshold=similarity_threshold, top_n=30)

        for i, item in enumerate(important, 1):
            report.append(f"\n{i}. [{item['frequency']} times] {item['question']}")
            report.append(f"   Appears in: {', '.join(item['appears_in'])}")

        # Key Topics
        report.append("\n\n" + "="*60)
        report.append("2. KEY TOPICS (by TF-IDF importance)")
        report.append("="*60)
        topics = self.extract_key_topics(top_n=30)

        for i, (topic, score) in enumerate(topics, 1):
            report.append(f"{i}. {topic} (importance: {score:.4f})")

        # Question Distribution
        report.append("\n\n" + "="*60)
        report.append("3. QUESTION DISTRIBUTION BY PAPER")
        report.append("="*60)
        source_counts = Counter(self.question_sources)
        for source, count in source_counts.most_common():
            report.append(f"{source}: {count} questions")

        # Save report
        report_text = '\n'.join(report)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n✓ Report saved to {output_file}")
        return report_text


# ============= MAIN USAGE =============

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ImportantQuestionsAnalyzer()

    # Option 1: Process all PDFs in a folder
    folder_path = "question_papers"  # Change this to your folder path

    if os.path.exists(folder_path):
        analyzer.process_folder(folder_path)
    else:
        print(f"Folder '{folder_path}' not found!")
        print("Creating example with individual PDFs...")

        # Option 2: Add individual PDFs
        pdf_files = ["nlp.pdf",
                     "nlp1.pdf",]
        for pdf in pdf_files:
            if os.path.exists(pdf):
                analyzer.add_questions_from_pdf(pdf)

    # Generate analysis report
    if len(analyzer.all_questions) > 0:
        report = analyzer.generate_report(
            output_file='important_questions_report.txt',
            similarity_threshold=0.65  # Adjust threshold (0.6-0.8 recommended)
        )

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nFiles generated:")
        print("  - important_questions_report.txt")
        print("\nThe report contains:")
        print("  1. Most frequently asked questions")
        print("  2. Key topics by importance")
        print("  3. Question distribution by paper")
    else:
        print("\nNo questions found. Please check your PDF files!")