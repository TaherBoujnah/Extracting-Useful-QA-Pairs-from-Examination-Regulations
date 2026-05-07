# 🎓 UniBot: RAG-based Academic Advisor & QA Extraction Pipeline

<video src="assets/demo1.mp4" width="100%" controls></video>

## 📌 Overview
This project is an end-to-end **Retrieval-Augmented Generation (RAG) system** designed to help university students navigate complex examination regulations (Prüfungsordnungen). 

Instead of just vectorizing raw PDFs, this project features a **sophisticated synthetic data generation pipeline** that uses LLMs to extract, evaluate, and filter High-Quality QA pairs from legal documents, resulting in a dual-tier RAG system.

## 🏗️ Architecture & Pipeline
The project runs a 9-step pipeline, transforming raw legal PDFs into a smart, conversational assistant:

1. **PDF Parsing:** Uses `LlamaParse` (via REST API) to convert dense academic PDFs into clean Markdown.
2. **Markdown Cleaning:** Rule-based Python scripts (`clean_markdown.py`) scrub out irrelevant sections (e.g., other degree programs) to prevent hallucinations.
3. **Context-Aware Chunking:** Deterministic legal chunking (`chunk_examregs.py`) bundles sections, paragraphs, and rules together.
4. **Semantic Filtering:** Uses **DSPy + Gemma 2 (2B)** to drop bureaucratic fluff and keep only actionable academic rules.
5. **Synthetic FAQ Generation:** Uses **DSPy + Gemma 3 (27B)** to generate realistic student questions and answers based strictly on the text.
6. **Semantic Deduplication:** Uses `sentence-transformers` (`intfloat/multilingual-e5-large`) to filter out semantically duplicate FAQs.
7. **LLM-as-a-Judge Validation:** Uses **DSPy + Llama 3.1 (8B)** as a strict university auditor to grade the generated FAQs (1-5 scale), dropping any hallucinations.
8. **Vector Database:** Compiles the surviving Gold FAQs and Raw Chunks into a dual-tier **ChromaDB** index.
9. **Streamlit UI:** A responsive chat interface (`app.py`) where students can ask questions and get cited answers.

## 🚀 Tech Stack
* **UI:** Streamlit
* **Orchestration:** DSPy
* **Local LLMs:** Ollama (Gemma 3:27b, Gemma 2:2b, Llama 3.1:8b)
* **Vector DB & Embeddings:** ChromaDB, Sentence Transformers
* **Document Parsing:** LlamaCloud / LlamaParse

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/TaherBoujnah/Extracting-Useful-QA-Pairs-from-Examination-Regulations.git](https://github.com/TaherBoujnah/Extracting-Useful-QA-Pairs-from-Examination-Regulations.git)
cd Extracting-Useful-QA-Pairs-from-Examination-Regulations



