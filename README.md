![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-purple)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow)
![Groq](https://img.shields.io/badge/LLM-Groq-red)
![RAG](https://img.shields.io/badge/AI-RAG-critical)
![HITL](https://img.shields.io/badge/System-HITL-blueviolet)


# 🚀 RAG-Based Customer Support Assistant

### (LangGraph + Human-in-the-Loop Escalation)

---

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based customer support assistant that answers user queries from a PDF knowledge base.

The system uses:

* Semantic search (embeddings + vector DB)
* LLM-based answer generation
* Graph-based workflow (LangGraph)
* Human-in-the-Loop (HITL) escalation for uncertain queries

---

## 🎯 Key Features

* 📄 Load and process PDF knowledge base
* 🔍 Semantic retrieval using embeddings
* 🤖 Context-aware answer generation (LLM)
* 🔀 Conditional routing using LangGraph
* 🚨 Human-in-the-loop escalation for low confidence
* 💻 CLI-based interactive interface

---

## 🧠 How It Works

```text
User Query
   ↓
LangGraph Workflow
   ↓
Retrieve Relevant Chunks (ChromaDB)
   ↓
Generate Answer (LLM)
   ↓
Confidence Check
   ↓
 ┌───────────────┬───────────────┐
 ↓               ↓
Answer       Escalation (HITL)
```

---

## 🏗️ Project Structure

```text
Rag_Support_Assistant/
│
├── main.py                # Entry point
├── rag_pipeline.py        # RAG logic (retrieval + generation)
├── graph_workflow.py      # LangGraph workflow (routing logic)
├── hitl.py                # Human escalation logic
├── utils.py               # Helper functions (logging, etc.)
│
├── knowledge_base.pdf     # Sample knowledge base
├── create_sample_pdf.py   # Script to generate sample PDF
│
├── chroma_db/             # Vector database (auto-generated)
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Tech Stack

* Python
* LangChain
* LangGraph
* ChromaDB
* HuggingFace Embeddings
* Groq LLM

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd Rag_Support_Assistant
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Set API Key (Groq)

```bash
set GROQ_API_KEY=your_api_key_here
```

---

## 📄 Generate Sample PDF (Optional)

```bash
python create_sample_pdf.py
```

---

## 🧱 Build Vector Store

```bash
python main.py --build
```

---

## ▶️ Run the Application

```bash
python main.py
```

---

## 🧪 Example Queries

### ✅ Relevant Query

```text
How do I return a product?
```

✔ System retrieves correct answer

---

### 🔄 Paraphrased Query

```text
How can I send back a product?
```

✔ Semantic retrieval works

---

### ❌ Irrelevant Query

```text
What is AI?
```

🚨 Escalation triggered (HITL)

---

## 🔀 Conditional Routing Logic

The system decides between answering and escalation based on:

* Confidence score (similarity threshold)
* Detection of uncertain responses (e.g., "I don't know")

```python
if confidence < threshold or "i don't know" in answer.lower():
    escalate = True
```

---

## 🚨 Human-in-the-Loop (HITL)

* Triggered when system is not confident
* Simulated using predefined response
* Ensures reliability of system

---

## ⚠️ Limitations

* Supports only single PDF
* CLI-based (no UI)
* No real human integration (simulated HITL)
* No conversation memory

---

## 🔮 Future Improvements

* Multi-document support
* Streamlit/Web UI
* Chat history (memory)
* Real-time human escalation
* Cloud deployment

---

## 🧠 Key Learnings

* RAG pipeline design
* Vector databases and embeddings
* LangGraph workflow orchestration
* Conditional routing logic
* Human-in-the-loop system design

---

## 📌 Conclusion

This project demonstrates a complete **RAG-based system** with retrieval, generation, decision-making, and escalation. It combines practical implementation with system design concepts, making it suitable for real-world customer support automation.

---

## 🙌 Author

**Sai B**
CSE Graduate | Data Science Enthusiast

---
