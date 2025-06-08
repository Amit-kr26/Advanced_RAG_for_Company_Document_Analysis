# Advanced RAG for Company Document Analysis

## Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) system designed specifically for analyzing company documents, such as annual reports in PDF format. It leverages the power of large language models from OpenAI and Gemini to provide accurate and contextually relevant answers to user queries about these documents. The system processes PDFs, extracts structured data like tables, and uses sophisticated prompt engineering to ensure precise and consistent outputs, making it ideal for enterprise-level document analysis.

## Key Features

- **Company Document Specialization** Tailored for annual reports, adept at parsing financial tables, business terminology, and multi-page structures.

- **Advanced Prompt Engineering** Leverages sophisticated prompts (see prompts.py) to rephrase questions, answer with RAG context, and enforce structured outputs (names, numbers, booleans, lists, comparisons).

- **Multi-Provider Flexibility**: Seamlessly integrates with OpenAI and Gemini APIs for robust language model support.

- **Table Serialization**: Transforms tables into context-independent blocks, enhancing tabular data comprehension.

- **Parallel Processing**: Boosts efficiency with asynchronous and parallel API request handling.

- **Structured Outputs**: Uses Pydantic models to ensure consistent, schema-driven responses for downstream use.

- **Robust Design**: Features retry logic, JSON repair, and error handling for reliability.

### Beyond Vanilla RAG

Unlike a standard RAG system that handles generic text corpora, thid project excels in:

- **Domain-Specific Parsing**: Tackles the intricate layouts of annual reports, including financial statements and tables.

- **Contextual Table Understanding**: Serializes tables with surrounding context for accurate interpretation.

- **Precision Answer Formats**: Delivers answers in strict schemas (e.g., exact numbers, full names) via prompts.py.

- **Enhanced Retrieval**: Combines vector similarity with optional reranking for superior relevance.

## Pipeline Overview

- **PDF Parsing**: Extracts text, tables, and images from PDFs (pdf_parsing.py).

- **Text Splitting**: Breaks text into chunks for embedding (text_splitter.py).

- **Table Serialization**: Converts tables into usable blocks (tables_serialization.py).

- **Vector Ingestion**: Stores embeddings in FAISS databases (ingestion.py).

- **Retrieval**: Fetches relevant chunks for queries (retrieval.py).

- **Reranking**: Refines results for accuracy (reranking.py).

- **Question Answering**: Generates precise answers using RAG context (questions_processing.py).

### Usage Example
``` python
from src.pipeline import Pipeline
from src.questions_processing import QuestionsProcessor

# Initialize and process PDFs
pipeline = Pipeline(root_path="path/to/pdfs")
pipeline.process_uploaded_pdfs()

# Ask a question
processor = QuestionsProcessor(vector_db_dir="path/to/vector_dbs", documents_dir="path/to/chunked_reports")
question = "What was Company X's revenue in 2022?"
answer = processor.process_question(question)
print(answer)
```

## File Descriptions
- ``__init__.py``: Initializes the package.

- ``api_request_parallel_processor.py``: Processes API requests concurrently while respecting rate limits.

- ``api_requests.py``: Manages interactions with OpenAI and Gemini APIs for embeddings and answers.

- ``ingestion.py``: Creates FAISS vector databases from chunked reports for retrieval.

- ``parsed_reports_merging.py``: Cleans and formats parsed report content into structured text.

- ``pdf_parsing.py``: Converts PDFs into JSON reports with text, tables, and metadata.

- ``pipeline.py``: Orchestrates the PDF processing and vector database creation workflow.

- ``prompts.py``: Defines advanced prompts and schemas for rephrasing questions and generating structured answers (e.g., RephrasedQuestionsPrompt, AnswerWithRAGContextNumberPrompt).

- ``questions_processing.py``: Answers queries using retrieved document chunks and RAG.

- ``reranking.py``: Reranks retrieved chunks for relevance using LLM-based scoring.

- ``retrieval.py``: Retrieves relevant chunks from vector databases using FAISS.

- ``tables_serialization.py``: Serializes tables into context-independent blocks with OpenAI.

- ``text_splitter.py``: Splits report text into chunks, preserving tables for embedding.
