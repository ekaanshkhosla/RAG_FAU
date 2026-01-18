# University-Specific Retrieval-Augmented Generation (RAG) System

## Overview

This project investigates the design and implementation of a **Retrieval-Augmented Generation (RAG)** system tailored for managing academic and administrative queries within the Department of Data Science at Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).

By integrating document retrieval with the generative capabilities of Large Language Models (LLMs), the system enables accurate, contextually grounded responses using both public and private university-related data.

## Web Interface

The screenshots below demonstrate the query-to-answer flow in our RAG-based chatbot.

<p align="center">
  <img src="User enters query.png" alt="User enters query" width="80%"><br>
  <span style="font-size: 120px;">⬇️</span><br>
  <img src="System generates answer.png" alt="System generates answer" width="80%">
</p>

**Figure:** The user enters a question (top), and the system responds with an answer (bottom) retrieved and generated using university documents.

## Motivation

The Department of Data Science receives a high volume of student queries related to:
- **Admissions** (e.g., deadlines, eligibility, scholarships)
- **Academics** (e.g., course selection, prerequisites, exam schedules)

Much of this information is fragmented across university portals, while other details are only available through private email exchanges with Student Advisory. This results in:
- Delays and frustration for students
- Increased administrative burden on staff

The RAG chatbot aims to address both problems by offering an intelligent, centralized solution.

## Objectives

- **Design** a RAG-based chatbot system that integrates public (websites, PDFs) and private (anonymized emails) data sources.
- **Develop** a retrieval pipeline that efficiently indexes and fetches relevant information using vector-based search and contrastive learning.
- **Deploy** a generative model (LLM) to generate accurate and coherent responses grounded in retrieved content.
- **Evaluate** system performance using both manual inspection and automated metrics such as [RAGAS](https://huggingface.co/spaces/RAGAS/ragas-space).

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ekaanshkhosla/RAG_FAU.git
cd RAG_FAU
```

### 2. Create and Activate Conda Environment
```bash
conda create -n rag_test python=3.11 -y
conda activate rag_test
```

### 3. Install the Application
```bash
pip install -e .
```

### 4. Run the Application
```bash
uvicorn app.main:app --reload
```

### 5. Open the application in your browser at:
```bash
http://127.0.0.1:8000/
```





