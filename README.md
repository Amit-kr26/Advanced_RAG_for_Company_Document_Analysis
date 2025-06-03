# Advanced RAG for Company Document Analysis

## Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) system designed specifically for analyzing company documents, such as annual reports in PDF format. It leverages the power of large language models from OpenAI and Gemini to provide accurate and contextually relevant answers to user queries about these documents. The system processes PDFs, extracts structured data like tables, and uses sophisticated prompt engineering to ensure precise and consistent outputs, making it ideal for enterprise-level document analysis.

## Key Features

``Table Serialization:`` Converts tables in PDFs into context-independent text blocks, enabling the system to understand and utilize structured data effectively. This is handled by the TableSerializer class, which processes tables into a format digestible by language models.



``Structured Prompts:`` Utilizes carefully crafted prompts and schemas defined in prompts.py to ensure that responses are consistent, parseable, and tailored to specific question types (e.g., names, numbers, booleans, lists, and comparative queries). This file is central to the system’s ability to handle company document queries with precision.



``Comparative Question Handling:`` Intelligently rephrases comparative questions (e.g., "Which company had higher revenue, 'Apple' or 'Microsoft'?") into individual queries for each company, enabling seamless multi-entity comparisons. This is driven by the RephrasedQuestionsPrompt in prompts.py.



``Efficient Retrieval:`` Employs FAISS vector databases with OpenAI embeddings for fast and accurate retrieval of relevant document chunks, with optional reranking to enhance precision.



``Asynchronous Processing:`` Supports parallel processing of multiple questions via AsyncOpenaiProcessor, significantly reducing response times for batch queries.



``Robustness:`` Includes comprehensive error handling, retry logic, and logging to ensure reliability in production environments.
