# OCR & RAG Architecture — Deep Dive

> **Project:** FYP Health App — ER Triage Clinical Assistant  
> **Date:** November 2026  
> **Scope:** Complete architectural breakdown of the OCR (Optical Character Recognition) and RAG (Retrieval-Augmented Generation) subsystems

---

## Table of Contents

1. [High-Level System Overview](#1-high-level-system-overview)
2. [OCR Subsystem — Architecture & Theory](#2-ocr-subsystem--architecture--theory)
   - 2.1 [What is OCR?](#21-what-is-ocr)
   - 2.2 [Why Gemini Multimodal Instead of Traditional OCR](#22-why-gemini-multimodal-instead-of-traditional-ocr)
   - 2.3 [OCR Pipeline Flow](#23-ocr-pipeline-flow)
   - 2.4 [Code Walkthrough](#24-code-walkthrough)
3. [RAG Subsystem — Architecture & Theory](#3-rag-subsystem--architecture--theory)
   - 3.1 [What is RAG?](#31-what-is-rag)
   - 3.2 [Why RAG Instead of Fine-Tuning?](#32-why-rag-instead-of-fine-tuning)
   - 3.3 [RAG Pipeline Flow](#33-rag-pipeline-flow)
   - 3.4 [Embeddings — The Core Theory](#34-embeddings--the-core-theory)
   - 3.5 [Vector Database — FAISS](#35-vector-database--faiss)
   - 3.6 [Ingestion Pipeline](#36-ingestion-pipeline)
   - 3.7 [Retrieval Pipeline](#37-retrieval-pipeline)
4. [OCR → RAG Integration — How They Connect](#4-ocr--rag-integration--how-they-connect)
5. [LangGraph Orchestration — The Agentic Flow](#5-langgraph-orchestration--the-agentic-flow)
6. [Technology Stack Summary](#6-technology-stack-summary)
7. [Key Theoretical Concepts for FYP Report](#7-key-theoretical-concepts-for-fyp-report)

---

## 1. High-Level System Overview

The system uses an **agentic architecture** powered by LangGraph, where the OCR and RAG modules act as processing nodes in a directed graph. The flow is:

```
User → Upload Image + Symptoms + Vitals
         │
         ▼
   ┌─────────────────────────────────────────────────┐
   │              LangGraph Agent Pipeline            │
   │                                                  │
   │  START                                           │
   │    │                                             │
   │    ▼                                             │
   │  [prepare_vitals]  ── ML Triage Prediction       │
   │    │                                             │
   │    ▼                                             │
   │  [ocr_node]  ── Gemini Multimodal OCR            │
   │    │                                             │
   │    ▼                                             │
   │  [retrieval_node]  ── FAISS Vector Search (RAG)  │
   │    │                                             │
   │    ▼                                             │
   │  [chatbot_node]  ── LLM Reasoning (Groq/LLaMA)  │
   │    │                                             │
   │    ▼ (conditional)                               │
   │  [tools] ←──→ [chatbot]  (Agentic Loop)         │
   │    │                                             │
   │    ▼                                             │
   │   END                                            │
   └─────────────────────────────────────────────────┘
```

The OCR extracts structured text from medical images. The RAG system uses that extracted text (combined with symptoms) to retrieve relevant medical guidelines from a vector database. Together, they give the LLM **grounded, evidence-based context** to reason with.

---

## 2. OCR Subsystem — Architecture & Theory

### 2.1 What is OCR?

**Optical Character Recognition (OCR)** is the process of converting images containing text (handwritten or printed) into machine-readable text. In our system, this is used for:

- **Prescriptions** — Extract medication names, dosages, instructions
- **Lab Reports** — Extract test names, values, reference ranges
- **Medical Documents** — Any text-based medical document

### 2.2 Why Gemini Multimodal Instead of Traditional OCR

| Approach | Traditional OCR (Tesseract, etc.) | Gemini Multimodal (Our Approach) |
|---|---|---|
| **How it works** | Pattern matching on character shapes | Visual understanding by a Vision-Language Model (VLM) |
| **Handwritten text** | Poor accuracy | Strong accuracy |
| **Medical jargon** | No context understanding | Understands medical terminology |
| **Structured output** | Raw text dump | Can organize by medications, dosages, etc. |
| **Noisy/blurry images** | Fails often | Handles gracefully due to visual reasoning |
| **Setup complexity** | Needs preprocessing pipelines | Single API call |

**Our system uses Google Gemini 1.5 Pro** — a multimodal large language model that can process both text and images simultaneously. Instead of doing character-by-character recognition, it *understands* the image holistically, which is critical for messy medical handwriting.

### 2.3 OCR Pipeline Flow

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User uploads│     │  PIL Image       │     │  Gemini 1.5 Pro │
│  image bytes ├────►│  object created  ├────►│  Multimodal API │
│  (JPEG/PNG)  │     │  from BytesIO    │     │  (Vision + Text)│
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Structured text │
                                              │  with medications│
                                              │  dosages, etc.   │
                                              └─────────────────┘
```

### 2.4 Code Walkthrough

**File: `backend/cv/image_model.py`**

```python
# 1. Initialize the Gemini client with API key
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# 2. The OCR function
def extract_text_from_image(image_bytes: bytes) -> str:
    # Convert raw bytes into a PIL Image object
    img = Image.open(BytesIO(image_bytes))
    
    # 3. The medical-specific prompt instructs Gemini to:
    #    - Read ALL text from the document
    #    - Structure it clearly
    #    - Highlight medications, dosages, conditions
    #    - NOT hallucinate anything
    prompt = (
        "You are a medical OCR assistant. Read this medical document..."
    )
    
    # 4. Send both the prompt AND the image to Gemini
    response = client.models.generate_content(
        model="gemini-1.5-pro",       # Multimodal model
        contents=[prompt, img]         # Text + Image together
    )
    
    return response.text  # Structured medical text
```

**Key design decisions:**
- **PIL Image object** is passed directly to the `google-genai` SDK (no base64 encoding needed)
- The **prompt engineering** is medical-specific — it tells Gemini to act as a "medical OCR assistant" and to never invent information
- **Error handling** returns meaningful messages instead of crashing the pipeline

---

## 3. RAG Subsystem — Architecture & Theory

### 3.1 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances LLM responses by retrieving relevant documents from an external knowledge base before generating an answer. Instead of relying solely on the LLM's training data (which may be outdated or generic), RAG provides **real-time, domain-specific context**.

```
Traditional LLM:
  Question ────► LLM (training data only) ────► Answer (may hallucinate)

RAG-Enhanced LLM:
  Question ────► Vector Search ────► Relevant Documents ────► LLM + Context ────► Grounded Answer
```

### 3.2 Why RAG Instead of Fine-Tuning?

| Approach | Fine-Tuning | RAG (Our Approach) |
|---|---|---|
| **Data update** | Retrain the entire model | Just add new PDFs to the vector store |
| **Cost** | Expensive GPU training | Cheap — only embedding + storage |
| **Hallucination** | Still possible | Reduced — LLM cites retrieved docs |
| **Transparency** | Black box | You can see what was retrieved |
| **Domain specificity** | Baked into weights | Dynamic — swap knowledge base anytime |

For a medical application, RAG is the better choice because:
1. Medical guidelines change frequently
2. We need to cite sources for patient safety
3. We don't have the resources to fine-tune a large model

### 3.3 RAG Pipeline Flow

The RAG system has two phases: **Ingestion** (offline, one-time) and **Retrieval** (online, per-query).

```
═══════════════════════════════════════════════════════════
  PHASE 1: INGESTION (Offline — Run Once)
═══════════════════════════════════════════════════════════

  ┌─────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────────────┐
  │  Medical │    │  PyPDFLoader │    │  Text Splitter│    │  HuggingFace  │
  │  PDFs    ├───►│  (extract    ├───►│  (chunk into  ├───►│  Embeddings   │
  │  (.pdf)  │    │   raw text)  │    │   1000 chars) │    │  (all-MiniLM) │
  └─────────┘    └──────────────┘    └───────────────┘    └──────┬────────┘
                                                                  │
                                                                  ▼
                                                         ┌───────────────┐
                                                         │  FAISS Vector │
                                                         │  Index (saved │
                                                         │  to disk)     │
                                                         └───────────────┘

═══════════════════════════════════════════════════════════
  PHASE 2: RETRIEVAL (Online — Per User Query)
═══════════════════════════════════════════════════════════

  ┌─────────────┐    ┌───────────────┐    ┌───────────────┐    ┌──────────┐
  │  symptoms + │    │  HuggingFace  │    │  FAISS        │    │  Top-3   │
  │  OCR text   ├───►│  Embeddings   ├───►│  Similarity   ├───►│  Matched │
  │  (query)    │    │  (same model) │    │  Search       │    │  Chunks  │
  └─────────────┘    └───────────────┘    └───────────────┘    └──────────┘
```

### 3.4 Embeddings — The Core Theory

**What are embeddings?**

An embedding is a **dense numerical vector** (a list of numbers) that represents the *semantic meaning* of a piece of text. Similar meanings produce similar vectors.

```
"patient has chest pain"     →  [0.23, -0.45, 0.67, 0.12, ...]  (384 dimensions)
"cardiac discomfort"         →  [0.21, -0.43, 0.65, 0.14, ...]  (very similar!)
"the weather is sunny"      →  [0.89, 0.12, -0.34, 0.56, ...]  (very different)
```

**Our Embedding Model: `all-MiniLM-L6-v2`**

| Property | Value |
|---|---|
| **Model Name** | `all-MiniLM-L6-v2` (from Sentence-Transformers / HuggingFace) |
| **Architecture** | MiniLM — a distilled version of BERT |
| **Output Dimensions** | 384-dimensional vectors |
| **Training Data** | 1B+ sentence pairs |
| **Speed** | ~14,000 sentences/sec on GPU |
| **Size** | ~80 MB (runs locally, no API needed) |
| **Why chosen** | Fast, lightweight, runs entirely offline — no API costs, no data leaves the machine |

**How it works mathematically:**

1. Text is tokenized into subword tokens
2. Tokens pass through 6 Transformer encoder layers (the "L6" in the name)
3. The [CLS] token's output is mean-pooled across all token positions
4. The result is a 384-dimensional unit vector

```
Input: "chest pain with shortness of breath"
  │
  ▼ Tokenize
["chest", "pain", "with", "short", "##ness", "of", "breath"]
  │
  ▼ 6 Transformer Layers
[hidden_1, hidden_2, ..., hidden_7]   (each is 384-dim)
  │
  ▼ Mean Pooling
single 384-dim vector
  │
  ▼ Normalize
[0.023, -0.145, 0.067, ...]   ← The embedding
```

### 3.5 Vector Database — FAISS

**What is FAISS?**

FAISS (**Facebook AI Similarity Search**) is an open-source library developed by Meta AI Research for efficient similarity search over dense vectors. It is the most widely used vector search library in the industry.

| Property | Value |
|---|---|
| **Full Name** | Facebook AI Similarity Search |
| **Developer** | Meta AI Research (formerly Facebook AI) |
| **Type** | Local vector index library (not a separate database server) |
| **Index Type Used** | `IndexFlatL2` (exact search, brute-force L2 distance) |
| **Storage** | Saved to disk as `index.faiss` + `index.pkl` |
| **Current Index Size** | `index.faiss` (1,581 bytes) + `index.pkl` (367 bytes) |
| **Search Metric** | L2 (Euclidean) distance → converted to similarity |
| **Why chosen** | No server needed, runs in-process, fast for small-to-medium datasets |

**How Similarity Search Works:**

When a query comes in, FAISS compares the query embedding against every stored embedding using **L2 (Euclidean) distance**:

$$d(q, d_i) = \sqrt{\sum_{j=1}^{384} (q_j - d_{i,j})^2}$$

Where:
- $q$ is the query embedding vector (384 dimensions)
- $d_i$ is the i-th document embedding vector
- Smaller distance = more similar

The top-k (in our case, k=3) most similar documents are returned.

**FAISS vs Other Vector Databases:**

| Feature | FAISS (Ours) | ChromaDB | Pinecone | Weaviate |
|---|---|---|---|---|
| **Deployment** | In-process library | Local server | Cloud SaaS | Self-hosted/Cloud |
| **Setup** | `pip install` | `pip install` | API key needed | Docker required |
| **Cost** | Free | Free | Paid | Free/Paid |
| **Persistence** | File on disk | SQLite | Cloud managed | Disk/Cloud |
| **Best for** | Small-medium datasets | Prototyping | Production scale | Enterprise |
| **Our reason** | Simple, no dependencies, perfect for FYP scope | — | — | — |

### 3.6 Ingestion Pipeline

**File: `backend/rag/ingest.py`**

The ingestion pipeline runs **once** (or whenever new medical PDFs are added) to populate the vector store.

```
Step 1: LOAD
┌─────────────────────────────────────────┐
│  PyPDFLoader reads each PDF in /data/   │
│  Extracts raw text page by page         │
│  Each page becomes a LangChain Document │
│  with metadata {source, page}           │
└─────────────────────────────────────────┘
              │
              ▼
Step 2: SPLIT (Chunking)
┌─────────────────────────────────────────┐
│  RecursiveCharacterTextSplitter         │
│                                         │
│  chunk_size = 1000 characters           │
│  chunk_overlap = 200 characters         │
│  separators = ["\n\n", "\n", " ", ""]   │
│                                         │
│  WHY: LLMs have token limits. We need   │
│  small, focused chunks for accurate     │
│  retrieval. Overlap ensures no context  │
│  is lost at chunk boundaries.           │
└─────────────────────────────────────────┘
              │
              ▼
Step 3: EMBED + STORE
┌─────────────────────────────────────────┐
│  Each chunk → all-MiniLM-L6-v2 → 384d  │
│  vector → stored in FAISS index         │
│                                         │
│  Saved to: backend/rag/faiss_index/     │
│    ├── index.faiss  (vector data)       │
│    └── index.pkl    (metadata + docs)   │
└─────────────────────────────────────────┘
```

**Why chunk_size=1000 and chunk_overlap=200?**

- **1000 characters** ≈ 200-250 tokens — small enough for focused retrieval, large enough to retain context
- **200 character overlap** — if a medication name spans across two chunks, the overlap ensures it appears in both, so retrieval won't miss it
- **Recursive separators** — tries to split at paragraph boundaries first (`\n\n`), then line breaks (`\n`), then spaces (` `), preserving natural text structure

### 3.7 Retrieval Pipeline

**File: `backend/rag/vectorstore.py`**

```python
def query_vectorstore(query_text: str, n_results: int = 3):
    vs = get_vectorstore()                            # Load FAISS index
    results = vs.similarity_search(query_text, k=3)   # Find 3 most similar chunks
    docs = [doc for doc in results if doc.page_content != "[System init]"]
    return docs
```

**What happens under the hood:**

```
1. query_text = "chest pain, shortness of breath, 
                  Metoprolol 50mg prescribed"
                              │
                              ▼
2. Embed query using same all-MiniLM-L6-v2 model
   → [0.023, -0.145, 0.067, ...]  (384-dim vector)
                              │
                              ▼
3. FAISS computes L2 distance to every stored vector
   doc_1: distance = 0.82  (cardiology guideline)     ← MATCH
   doc_2: distance = 0.91  (emergency triage protocol) ← MATCH
   doc_3: distance = 0.95  (Metoprolol drug info)      ← MATCH
   doc_4: distance = 1.78  (dermatology guideline)     ← too far
   ...
                              │
                              ▼
4. Return top-3 most similar document chunks
```

---

## 4. OCR → RAG Integration — How They Connect

This is the critical integration point. The OCR output doesn't just sit in state — it **enriches the RAG query**.

**File: `backend/agent/nodes/retrieval_node.py`**

```python
def retrieval_node(state: AgentState) -> dict:
    # Start with the user's symptoms as the query
    query_text = state.get("symptoms_text", "")
    
    # If OCR extracted text from an uploaded image, APPEND it
    ocr_text = state.get("ocr_text")
    if ocr_text:
        query_text += f"\n\nContext from scanned document:\n{ocr_text}"
    
    # Now search FAISS with the combined query
    docs = query_vectorstore(query_text, n_results=3)
    
    return {"retrieved_context": [doc.page_content for doc in docs]}
```

**Why this matters:**

Without OCR integration, the RAG query would be:
> *"I have chest pain"*  
> → Retrieves generic chest pain guidelines

With OCR integration, the RAG query becomes:
> *"I have chest pain*  
> *Context from scanned document:*  
> *Medication: Metoprolol 50mg, twice daily*  
> *Diagnosis: Hypertensive cardiomyopathy"*  
> → Retrieves guidelines specific to Metoprolol interactions + cardiomyopathy management

The OCR text adds **clinical specificity** to the vector search, pulling back far more relevant medical guidelines.

---

## 5. LangGraph Orchestration — The Agentic Flow

The OCR and RAG modules are orchestrated by **LangGraph** — a library for building stateful, multi-step AI agent workflows as directed graphs.

**File: `backend/agent/graph.py`**

```
                    ┌─────────────┐
                    │    START     │
                    └──────┬──────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ prepare_vitals  │  ← ML Model predicts triage level
                  │ (Random Forest) │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   ocr_node      │  ← Gemini 1.5 Pro extracts text
                  │ (Gemini Vision) │     from uploaded prescription/report
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  retrieval_node │  ← FAISS similarity search
                  │  (RAG / FAISS)  │     query = symptoms + OCR text
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
              ┌──►│  chatbot_node   │  ← Groq LLaMA-3 70B reasons over
              │   │  (Groq/LLaMA)   │     all collected context
              │   └────────┬────────┘
              │            │
              │     ┌──────┴──────┐
              │     │ Has tool    │
              │     │ calls?      │
              │     └──┬──────┬───┘
              │    YES │      │ NO
              │        ▼      ▼
              │   ┌────────┐ ┌─────┐
              └───┤ tools  │ │ END │
                  └────────┘ └─────┘
```

**State Flow (what data flows through):**

```
AgentState = {
    # Inputs (set by routes_chat.py)
    symptoms_text:     "chest pain, dizziness"
    image_bytes:       <raw bytes of uploaded image>
    vitals:            {heart_rate: 95, bp: 140/90, ...}
    user_location:     "Lahore, Punjab, Pakistan"
    
    # Filled by prepare_vitals node
    ml_prediction:     {triage_label: "Emergent", probability: 87%}
    
    # Filled by ocr_node
    ocr_text:          "Rx: Metoprolol 50mg BD, Aspirin 75mg OD..."
    
    # Filled by retrieval_node
    retrieved_context: ["Guideline chunk 1...", "Chunk 2...", "Chunk 3..."]
    
    # Filled by chatbot_node
    messages:          [SystemMessage, HumanMessage, AIMessage, ...]
}
```

---

## 6. Technology Stack Summary

| Layer | Technology | Purpose |
|---|---|---|
| **OCR Engine** | Google Gemini 1.5 Pro (Multimodal) | Vision-Language model reads medical images |
| **Embedding Model** | `all-MiniLM-L6-v2` (HuggingFace) | Converts text → 384-dim semantic vectors |
| **Vector Database** | FAISS (Meta AI) | Stores and searches embeddings locally |
| **Document Loader** | PyPDFLoader (LangChain) | Extracts text from medical PDF guidelines |
| **Text Splitter** | RecursiveCharacterTextSplitter | Chunks documents (1000 chars, 200 overlap) |
| **Agent Framework** | LangGraph | Orchestrates the multi-node processing pipeline |
| **Chat LLM** | Groq Cloud — LLaMA-3 70B | Final reasoning and response generation |
| **API Framework** | FastAPI | REST API serving the frontend |
| **SDK** | `google-genai` (Gemini), `langchain` | AI/ML integration libraries |

---

## 7. Key Theoretical Concepts for FYP Report

### 7.1 Retrieval-Augmented Generation (RAG)

RAG was introduced by Lewis et al. (2020) in the paper *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*. The key insight is that LLMs can be made more factual and up-to-date by providing them with retrieved documents as additional context, rather than relying solely on parametric knowledge learned during pre-training.

**The RAG formula (conceptual):**

$$P(y|x) = \sum_{d \in \text{top-k}} P(d|x) \cdot P(y|x, d)$$

Where:
- $x$ = input query (symptoms + OCR text)
- $d$ = retrieved document chunk
- $y$ = generated response
- $P(d|x)$ = retrieval probability (how relevant is this document?)
- $P(y|x, d)$ = generation probability (LLM output given query + document)

### 7.2 Semantic Similarity Search

Unlike keyword search (BM25/TF-IDF) which matches exact words, **semantic search** matches meaning. This is critical in medicine where:
- "MI" and "myocardial infarction" and "heart attack" are the same thing
- "SOB" means "shortness of breath" (not what it usually means!)
- "Rx" means "prescription"

The embedding model captures these semantic relationships because it was trained on billions of sentence pairs.

### 7.3 Multimodal Vision-Language Models (VLMs)

Traditional OCR (like Tesseract) uses:
1. Image preprocessing (binarization, deskewing)
2. Character segmentation
3. Pattern matching against known character templates

Gemini 1.5 Pro instead uses a **Vision Transformer (ViT)** architecture:
1. Image is split into patches (16×16 pixels)
2. Each patch is linearly projected into an embedding
3. These visual embeddings are processed alongside text token embeddings
4. The model attends to both modalities simultaneously via cross-attention

This allows Gemini to understand **layout**, **context**, and **medical terminology** — not just individual characters.

### 7.4 Chunking Strategy

Our system uses **RecursiveCharacterTextSplitter** with these parameters:

- **chunk_size = 1000**: Each chunk is ≈200-250 tokens. Research suggests chunks of 100-500 tokens work best for retrieval — small enough to be topically focused, large enough to carry meaningful context.
- **chunk_overlap = 200**: 20% overlap prevents information loss at chunk boundaries. If a drug interaction warning spans two paragraphs, both chunks will contain the full warning.
- **Recursive separators** `["\n\n", "\n", " ", ""]`: Tries to split at natural boundaries first, preserving paragraph and sentence structure.

### 7.5 FAISS Indexing

Our system uses `IndexFlatL2` — the simplest FAISS index type:

- **Exact search**: Compares the query against every stored vector (no approximation)
- **L2 distance**: Euclidean distance between vectors
- **Time complexity**: O(n × d) where n = number of documents, d = 384 dimensions
- **Perfect for our scale**: With hundreds to low thousands of medical guideline chunks, exact search is fast enough (<10ms)

For larger-scale production systems, FAISS offers approximate search indices (IVF, HNSW) that trade small accuracy losses for 10-100x speed improvements.

---

> **Note:** This architecture document covers the system as implemented. The OCR and RAG pipelines are fully functional and integrated into the LangGraph agent workflow. No additional development is needed for the core OCR → RAG flow.

