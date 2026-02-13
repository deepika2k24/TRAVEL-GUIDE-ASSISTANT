# PAYANA – RAG-Based Travel Assistant

An AI-powered travel planning assistant built using Retrieval-Augmented Generation (RAG).  
This system generates personalized Karnataka travel itineraries based on user queries, budget, and number of days.

---

## 🚀 Project Overview

This project combines:

- FAISS Vector Search for semantic retrieval  
- Sentence Transformers for embeddings  
- Groq LLM (Llama 3.1 8B Instant) for itinerary generation  
- ScaleDown API for context compression  
- Flask Backend with CORS support  
- Unsplash API for destination images  

Architecture Flow:

User Query  
→ Embedding (SentenceTransformer)  
→ FAISS Vector Search  
→ Context Compression (ScaleDown)  
→ LLM Generation (Groq)  
→ Structured JSON Travel Plan  

---

## ✨ Features

- Semantic search over Karnataka tourism data  
- Budget-aware itinerary generation  
- Dynamic day extraction from user query  
- Clean JSON enforcement with auto-repair parsing  
- Tourism-focused summarization  
- Image enrichment using Unsplash API  
- Production-ready Flask API  

---

## 📦 Tech Stack

- Python 3.10+
- Flask
- FAISS (CPU)
- SentenceTransformers (`all-MiniLM-L6-v2`)
- Groq API (Llama 3.1 8B Instant)
- ScaleDown API
- Unsplash API

---

---

## 🔑 Environment Variables

Create a `.env` file inside the backend folder:

```
GROQ_API_KEY=your_groq_api_key
SCALEDOWN_API_KEY=your_scaledown_api_key
UNSPLASH_KEY=your_unsplash_key
```
## ⚙️ Installation
```
git clone https://github.com/yourusername/karnataka-rag-travel.git
cd backend
pip install flask flask-cors faiss-cpu numpy requests sentence-transformers python-dotenv
```
## ▶️ Running the Server
```
python app.py
```
## Server runs at:
```
http://localhost:5000
```

## DEMO VIDEO:



https://github.com/user-attachments/assets/bfee0769-af3b-4b25-b55c-1e28466a54ef


