# CRAG for Indian Legal Question Answering

A Corrective Retrieval-Augmented Generation (CRAG) system for answering statute-based legal questions in the Indian legal domain, with comparison against standard RAG.

---

## 📌 Overview

This project builds a **Retrieval-Augmented Generation (RAG)** system and extends it to **Corrective RAG (CRAG)** by introducing:

- Query rewriting  
- Retrieval evaluation  
- Multi-hop retrieval  
- Local LLM-based answer generation  
- Local evaluation (groundedness + relevance)  

The system answers queries such as:
- "What is cheating under IPC?"
- "What is FIR procedure?"
- "Cheating vs cheating by personation?"

---

## 🧱 Corpus Construction

### Datasets Used

The legal corpus was constructed using publicly available structured datasets:

- Constitution of India dataset (GitHub)  
- Civic Tech India datasets covering:
  - Indian Penal Code (IPC)  
  - Code of Criminal Procedure (CRPC)  
  - Code of Civil Procedure (CPC)  
  - Indian Evidence Act (IEA)  

---

### Why These Datasets?

These datasets were chosen because:

- They provide **clean, structured statutory text** (no OCR noise)  
- They cover **core Indian legal domains** required for QA  
- They are **open-source and reproducible**  
- They are already **section-wise segmented**, aligning with retrieval tasks  

This avoids the need for heavy preprocessing and improves retrieval quality.

---

### Corpus Structure

Each legal provision is stored in a structured JSON format:

```json
{
  "doc_type": "statute",
  "source": "CRPC",
  "section": "126",
  "article": null,
  "clause": null,
  "sub_clause": null,
  "title": "Procedure",
  "text": "...",
  "citation": "Section 126 CRPC"
}
```

---

### Why This Structure?

The schema is designed to support both **retrieval performance** and **legal interpretability**:

- **doc_type** → allows future extension (e.g., case law)  
- **source** → enables domain-aware retrieval (IPC / CRPC / IEA / COI)  
- **section / article / clause** → preserves legal hierarchy  
- **title** → strong semantic signal for embeddings  
- **text** → full content used for retrieval and generation  
- **citation** → standardized reference for evaluation and answers  

---

### Benefits

- Improves embedding quality (title + content)  
- Enables precise citation matching  
- Supports reranking effectively  
- Easily extensible for future legal datasets  


## ⚙️ System Architecture

```
Query
↓
Retriever (FAISS + embeddings)
↓
Reranker (CrossEncoder)
↓
Evaluator (confidence check)
↓
Corrector (query rewrite)
↓
Retriever (again)
↓
Generator (Rule-based / LLM)
↓
Answer
```

---

## 🧠 Key Components

### 1. Retriever (`retriever.py`)
- Dense retrieval using SentenceTransformers  
- FAISS index  
- Cross-encoder reranking  
- Source inference (IPC / CRPC / IEA / COI)  
- Definition-aware boosting  

---

### 2. CRAG Pipeline (`pipeline.py`)
- Iterative retrieval  
- Query correction  
- Best-result tracking  
- Supports:
  - Rule-based generation  
  - LLM-based generation (Ollama)  

---

### 3. Generator (`generator.py`)

#### Rule-based Generator
- Selects most relevant section  
- Prioritizes definition sections  
- Deterministic output  

#### LLM Generator (Ollama)
- Uses local LLM (LLaMA / Mistral)  
- Context-grounded responses  
- Prompt-based generation  

---

### 4. Corrector (`corrector.py`)
- Expands abbreviations (IPC, CRPC, etc.)  
- Adds intent-based hints  
- Improves retrieval queries  

---

### 5. Evaluator (`evaluator.py`)
- Computes semantic similarity  
- Detects noisy documents  
- Guides CRAG correction loop  

---

### 6. Local LLM Evaluator (`local_evaluator.py`)
- Groundedness scoring  
- Relevance scoring  
- Uses local LLM (Ollama)  
- No external APIs required  

---

## 📊 Evaluation

We evaluate the system at multiple levels:

### 🔹 1. Retrieval-Level Metrics (`evaluate.py`)
- Accuracy@1  
- Hit@K  
- Mean Reciprocal Rank (MRR)  

### 🔹 2. Answer-Level Metrics (`evaluate_with_generator.py`)
- Final answer accuracy  
- Citation correctness  

### 🔹 3. LLM-Based Metrics (`evaluate_llm_local.py`)
- Groundedness (is answer supported by context?)  
- Relevance (does answer match query?)  

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install faiss-cpu sentence-transformers requests
```

### 3. Setup Ollama (for LLM generation & evaluation)

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull a model:

```bash
ollama pull llama3
# or
ollama pull mistral
```

Run:

```bash
ollama serve
```

---

## ▶️ Running the System

### Run CRAG pipeline

```bash
python pipeline.py
```

### Retrieval Evaluation

```bash
python evaluate.py
```

### Answer Evaluation

```bash
python evaluate_with_generator.py
```

### LLM-based Evaluation

```bash
python evaluate_llm_local.py
```

⚠️ Recommended: run on **20–30 queries** due to LLM latency.

---

## 📂 Dataset Format

```json
[
  {
    "query": "What is cheating under IPC?",
    "gold_citations": ["Section 415 IPC"]
  }
]
```

---

## 🔍 Key Observations

- CRAG improves performance on:
  - noisy queries  
  - ambiguous queries  
  - incorrect domain queries  

- CRAG may:
  - produce broader answers  
  - reduce precision in some cases  

- Query rewriting affects:
  - retrieval quality  
  - generation behavior  

---

## ⚠️ Limitations

- Small dataset size  
- Local LLM variability  
- Strict evaluation metrics (string matching)  
- Over-generation in CRAG  

---

## 🔮 Future Work

- Better query classification (definition vs comparison)  
- Multi-hop reasoning  
- Larger and more realistic dataset  
- Stronger LLMs  
- Improved evaluation metrics (precision/recall)  

---

## 🧠 Key Insight

Query rewriting improves retrieval, but must be decoupled from generation to avoid altering the intended question semantics.

---

## 👥 Contributors

- <Name 1>  
- <Name 2>  
- <Name 3>  

---

## 📄 License

MIT License (or specify your own)
