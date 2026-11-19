# NexaCare — Complete Project Architecture (Interview Prep)

> **Project:** NexaCare — Next-Gen AI-Powered Healthcare App  
> **Role:** AI Engineer / Full-Stack Developer  
> **Date:** November 2026  
> **Prep For:** Capgemini Interview

---

## Table of Contents

1. [One-Liner Project Summary](#1-one-liner-project-summary)
2. [System Architecture — Big Picture](#2-system-architecture--big-picture)
3. [The 6 AI/ML Subsystems](#3-the-6-aiml-subsystems)
4. [LangGraph StateGraph — Deep Dive](#4-langgraph-stategraph--deep-dive)
5. [AgentState — The Central Data Bus](#5-agentstate--the-central-data-bus)
6. [Node-by-Node Walkthrough](#6-node-by-node-walkthrough)
7. [The Agentic Tool Loop](#7-the-agentic-tool-loop)
8. [RAG Architecture — Theory & Implementation](#8-rag-architecture--theory--implementation)
9. [OCR Architecture — Theory & Implementation](#9-ocr-architecture--theory--implementation)
10. [Database & Auth Architecture](#10-database--auth-architecture)
11. [API Architecture](#11-api-architecture)
12. [Frontend Architecture](#12-frontend-architecture)
13. [Interview Q&A — 25 Questions They Will Ask](#13-interview-qa--25-questions-they-will-ask)

---

## 1. One-Liner Project Summary

> **NexaCare** is an agentic clinical decision-support system that combines ML triage prediction, deep learning brain tumor detection, multimodal OCR, RAG-powered medical knowledge retrieval, and LLM reasoning — orchestrated as a LangGraph stateful agent with tool-calling capabilities.

**When they ask "Tell me about your project":**

> "I built an AI-powered ER triage assistant. A patient enters their symptoms and vitals — a Random Forest model predicts triage urgency, a FAISS-backed RAG system retrieves relevant medical guidelines, and if they upload a prescription, Gemini's vision model extracts the text. All of this context flows through a LangGraph agent where LLaMA-3 70B reasons over everything and can autonomously call tools to find nearby hospitals and pharmacies. The system also does brain tumor classification from MRI scans using an ONNX deep learning model, and sends automated email summaries after every visit."

---

## 2. System Architecture — Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (HTML/CSS/JS)                       │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐ │
│  │  Login/   │ │  Manual   │ │  Tumor   │ │Dashboard│ │ Hospital │ │
│  │  Register │ │  Entry    │ │  Mode    │ │& Profile│ │ Locator  │ │
│  └────┬─────┘ └─────┬─────┘ └────┬─────┘ └────┬────┘ └────┬─────┘ │
└───────┼─────────────┼────────────┼─────────────┼───────────┼───────┘
        │             │            │             │           │
        ▼             ▼            ▼             ▼           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (REST API)                       │
│                                                                     │
│  POST /auth/register    POST /auth/login                            │
│  POST /chat/            POST /predict/     POST /predict/mri        │
│  GET  /user/history     GET  /user/locator                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐  ┌───────────────────┐  ┌───────────────────────┐
│   ML Pipeline   │  │  LangGraph Agent  │  │   DL Pipeline         │
│  (Random Forest)│  │  (Stateful Graph) │  │  (ONNX Brain Tumor)   │
│  pipe.pkl       │  │                   │  │  brain_tumor_model.onnx│
│                 │  │  5 Nodes:         │  │                       │
│  Input: Vitals  │  │  prepare_vitals   │  │  Input: MRI Image     │
│  Output: Triage │  │  ocr              │  │  Output: Glioma/      │
│  Level 0-3      │  │  retrieval (RAG)  │  │  Meningioma/Pituitary/│
│                 │  │  chatbot (LLM)    │  │  No Tumor             │
│                 │  │  tools            │  │                       │
└─────────────────┘  └────────┬──────────┘  └───────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │  Gemini 1.5  │ │  FAISS       │ │  Groq Cloud  │
     │  Pro (OCR)   │ │  Vector DB   │ │  LLaMA-3 70B │
     │  Google API  │ │  (Local)     │ │  (Reasoning)  │
     └──────────────┘ └──────────────┘ └──────────────┘
              │               │               │
              │               │               ▼
              │               │        ┌──────────────┐
              │               │        │  Tools:       │
              │               │        │  • Hospital   │
              │               │        │    Locator    │
              │               │        │  • Pharmacy   │
              │               │        │    Locator    │
              │               │        │  • Email Alert│
              │               │        └──────────────┘
              │               │
              ▼               ▼
     ┌────────────────────────────────┐
     │        SQLite Database         │
     │  Users │ Profiles │ Visits     │
     │  MRI Predictions               │
     └────────────────────────────────┘
```

---

## 3. The 6 AI/ML Subsystems

| # | Subsystem | Model | Type | Input | Output |
|---|---|---|---|---|---|
| 1 | **ML Triage** | Random Forest (pipe.pkl) | Classical ML | 9 vital signs | Triage Level 0-3 + probability |
| 2 | **DL Tumor Detection** | Custom CNN (ONNX) | Deep Learning | MRI Image (150×150) | Glioma/Meningioma/Pituitary/No Tumor |
| 3 | **OCR** | Gemini 1.5 Pro | Vision-Language Model | Prescription/Report Image | Structured medical text |
| 4 | **RAG** | all-MiniLM-L6-v2 + FAISS | Embedding + Vector Search | Symptoms + OCR text | Top-3 relevant medical guideline chunks |
| 5 | **LLM Reasoning** | LLaMA-3 70B (via Groq) | Large Language Model | All context combined | Clinical advice response |
| 6 | **Tool Calling** | LLM Function Calling | Agentic AI | Location + severity | Hospital/Pharmacy results |

---

## 4. LangGraph StateGraph — Deep Dive

### What is LangGraph?

LangGraph is a framework by LangChain for building **stateful, multi-step AI agent workflows** as directed graphs. Unlike simple prompt-response LLM calls, LangGraph lets you:

1. **Define nodes** — Each node is a function that processes/transforms data
2. **Define edges** — Connections between nodes (linear or conditional)
3. **Maintain state** — A shared `AgentState` dictionary flows through all nodes
4. **Add memory** — `MemorySaver` preserves conversation history across API calls
5. **Enable tool use** — The LLM can autonomously decide to call external tools

### Why StateGraph (Not a Simple Chain)?

```
SIMPLE CHAIN (LangChain):
  Input → LLM → Output
  (one shot, no branching, no tools, no memory)

STATE GRAPH (LangGraph):
  Input → Node1 → Node2 → Node3 → LLM → [Decision]
                                            ├─ Call Tool → Loop back to LLM
                                            └─ No Tool  → END
  (stateful, branching, tools, memory, multi-turn)
```

A StateGraph is essentially a **finite state machine** where:
- **States** = nodes (functions that transform data)
- **Transitions** = edges (connections between nodes)
- **Context** = AgentState (shared data that all nodes can read/write)

### The Graph Structure

```python
# File: backend/agent/graph.py

workflow = StateGraph(AgentState)        # 1. Create graph with state schema

workflow.add_node("prepare_vitals", ...)  # 2. Add nodes
workflow.add_node("ocr", ...)
workflow.add_node("retrieval", ...)
workflow.add_node("chatbot", ...)
workflow.add_node("tools", ToolNode(...))

workflow.add_edge(START, "prepare_vitals") # 3. Wire edges
workflow.add_edge("prepare_vitals", "ocr")
workflow.add_edge("ocr", "retrieval")
workflow.add_edge("retrieval", "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)  # 4. Conditional routing
workflow.add_edge("tools", "chatbot")     # 5. Tool loop

memory = MemorySaver()                    # 6. Add conversation memory
clinical_agent = workflow.compile(checkpointer=memory)  # 7. Compile
```

### Visual Graph:

```
                    ┌─────────┐
                    │  START   │
                    └────┬────┘
                         │
                         ▼
              ┌────────────────────┐
              │  prepare_vitals    │  Node 1: ML Triage Prediction
              │  (Random Forest)   │  Input:  vitals from user
              │                    │  Output: {ml_prediction: {level, label, probability}}
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  ocr               │  Node 2: Medical Image OCR
              │  (Gemini 1.5 Pro)  │  Input:  image_bytes from upload
              │                    │  Output: {ocr_text: "Rx: Metoprolol..."}
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  retrieval         │  Node 3: RAG Vector Search
              │  (FAISS + MiniLM)  │  Input:  symptoms + ocr_text
              │                    │  Output: {retrieved_context: ["chunk1", ...]}
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
          ┌──►│  chatbot           │  Node 4: LLM Reasoning
          │   │  (Groq LLaMA-3)   │  Input:  ALL state (vitals, OCR, RAG, history)
          │   │                    │  Output: {messages: [AIMessage(...)]}
          │   └─────────┬──────────┘
          │             │
          │      ┌──────┴──────┐
          │      │ tools_       │   Conditional Edge:
          │      │ condition    │   Does the LLM want to call a tool?
          │      └──┬───────┬──┘
          │     YES │       │ NO
          │         ▼       ▼
          │   ┌──────────┐ ┌─────┐
          │   │  tools   │ │ END │
          │   │  (auto-  │ └─────┘
          └───┤  execute)│
              └──────────┘
              Node 5: Runs hospital_locator
              and pharmacy_locator, then
              loops back to chatbot
```

---

## 5. AgentState — The Central Data Bus

The `AgentState` is a TypedDict that acts as the **shared memory** across all nodes. Every node reads from it and writes back to it.

```python
# File: backend/agent/state.py

class AgentState(TypedDict):
    # === CONVERSATIONAL MEMORY ===
    messages: Annotated[list[BaseMessage], add_messages]
    # ↑ Uses a REDUCER function: new messages are APPENDED, never replaced
    #   This is how chat history accumulates across turns
    
    # === USER INPUTS (set once by routes_chat.py) ===
    vitals: Optional[PatientVitals]      # 9 vital signs
    symptoms_text: str                    # "chest pain, dizziness"
    image_bytes: Optional[bytes]          # Raw uploaded image
    user_location: str                    # "Mumbai, Maharashtra"
    user_email: Optional[str]             # For email alerts
    
    # === PATIENT CONTEXT (loaded from database) ===
    patient_profile: Optional[dict]       # {name, age, gender, chronic_conditions}
    visit_history: List[dict]             # Last visits from DB
    
    # === INTERMEDIATE RESULTS (filled by nodes) ===
    ocr_text: Optional[str]              # Filled by ocr_node
    ml_prediction: Optional[PredictionResponse]  # Filled by prepare_vitals
    retrieved_context: List[str]          # Filled by retrieval_node
```

### The Reducer Pattern (Important Concept!)

```python
messages: Annotated[list[BaseMessage], add_messages]
```

The `add_messages` reducer means: when a node returns `{"messages": [new_msg]}`, it **appends** `new_msg` to the existing list instead of replacing it. This is how conversation history builds up:

```
Turn 1: messages = [HumanMessage("I have chest pain")]
Turn 2: messages = [HumanMessage("I have chest pain"), AIMessage("Based on...")]
Turn 3: messages = [HumanMessage("I have chest pain"), AIMessage("Based on..."), 
                     HumanMessage("What about medications?")]
```

---

## 6. Node-by-Node Walkthrough

### Node 1: prepare_vitals (ML Prediction)

```
Input:  state.vitals = {age: 55, heart_rate: 110, systolic_bp: 85, ...}
Model:  Random Forest (pipe.pkl) — trained on ESI triage dataset
Output: state.ml_prediction = {triage_level: 3, 
                                triage_label: "Level 3 - Emergent",
                                probability: 87.5}

Features used (9 total):
  age, heart_rate, systolic_blood_pressure, oxygen_saturation,
  body_temperature, pain_level, chronic_disease_count,
  previous_er_visits, arrival_mode
```

### Node 2: ocr (Gemini Multimodal OCR)

```
Input:  state.image_bytes = <raw JPEG/PNG bytes>
Model:  Google Gemini 1.5 Pro (Vision-Language Model)
Prompt: "You are a medical OCR assistant. Read this medical document.
         Extract all text, highlight medications, dosages, conditions.
         Do not invent any information."
Output: state.ocr_text = "Rx: Metoprolol 50mg twice daily, 
                          Aspirin 75mg once daily.
                          Diagnosis: Hypertensive cardiomyopathy"
```

### Node 3: retrieval (RAG)

```
Input:  query = state.symptoms_text + state.ocr_text (combined)
Model:  all-MiniLM-L6-v2 → 384-dim embedding → FAISS similarity search
Source: 3 medical PDFs (469 chunks in FAISS):
        - ESI Triage Handbook
        - WHO ETAT Guidelines
        - Drug Reference
Output: state.retrieved_context = [
          "ESI Level 3: Patients presenting with acute chest pain...",
          "Metoprolol: Beta-blocker used for hypertension...",
          "Triage protocol for cardiovascular emergencies..."
        ]
```

### Node 4: chatbot (LLM Reasoning)

```
Input:  EVERYTHING — builds a rich system prompt containing:
        - Patient profile (name, age, gender, chronic conditions)
        - Current symptoms + location
        - OCR text from scanned document
        - Past visit history (last 3 visits)
        - ML triage prediction + probability
        - RAG-retrieved clinical guidelines
Model:  Groq Cloud — LLaMA-3 70B (temperature=0 for deterministic output)
Output: Either:
        (a) Final answer → messages = [AIMessage("Based on your symptoms...")]
        (b) Tool call → messages = [AIMessage(tool_calls=[{hospital_locator, ...}])]
```

### Node 5: tools (Automatic Tool Execution)

```
Available Tools:
  1. hospital_locator(location, severity)
     → Google Places API (primary) + OpenStreetMap (fallback)
     → Returns 5 nearest hospitals with name, address, rating, phone, map link
  
  2. pharmacy_locator(medicine_name, location)  
     → Same hybrid API engine
     → Returns 5 nearest pharmacies

How it works:
  LangGraph's ToolNode automatically:
  1. Reads the tool_calls from the LLM's response
  2. Executes each tool function
  3. Returns results as ToolMessage objects
  4. Routes back to chatbot node (so LLM can read results)
```

---

## 7. The Agentic Tool Loop

This is the most interview-impressive part. The chatbot doesn't just answer — it **decides** whether to use tools.

```
┌──────────────────────────────────────────────────────┐
│  chatbot_node runs LLaMA-3 70B                       │
│                                                      │
│  LLM receives: system prompt + all context + history │
│                                                      │
│  LLM decides:                                        │
│  "I need to find hospitals near Mumbai for this      │
│   emergent patient, and pharmacies for Metoprolol"   │
│                                                      │
│  LLM outputs:                                        │
│  AIMessage(                                          │
│    tool_calls=[                                      │
│      {name: "hospital_locator",                      │
│       args: {location: "Mumbai", severity: "Emergent"}}, │
│      {name: "pharmacy_locator",                      │
│       args: {medicine_name: "Metoprolol",            │
│              location: "Mumbai"}}                    │
│    ]                                                 │
│  )                                                   │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼  tools_condition detects tool_calls → routes to "tools"
┌──────────────────────────────────────────────────────┐
│  ToolNode executes both tools in parallel:           │
│                                                      │
│  hospital_locator("Mumbai", "Emergent"):             │
│    → Google Places API → 5 trauma centers near Mumbai│
│    (falls back to OpenStreetMap if Google fails)     │
│                                                      │
│  pharmacy_locator("Metoprolol", "Mumbai"):           │
│    → Google Places API → 5 pharmacies near Mumbai    │
│                                                      │
│  Returns: [ToolMessage(hospital results),            │
│            ToolMessage(pharmacy results)]            │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼  Always loops back to chatbot
┌──────────────────────────────────────────────────────┐
│  chatbot_node runs AGAIN                             │
│                                                      │
│  Now the LLM sees the tool results in its messages   │
│  and writes a final human-readable response:         │
│                                                      │
│  "Based on your symptoms and triage level (Emergent),│
│   I recommend going to the nearest hospital          │
│   immediately.                                       │
│                                                      │
│   🏥 Nearest Hospitals:                              │
│   1. Lilavati Hospital - 📍 Bandra West...          │
│   2. Kokilaben Hospital - 📍 Andheri West...        │
│                                                      │
│   💊 Nearest Pharmacies:                             │
│   1. MedPlus - 📍 ..."                              │
│                                                      │
│  tools_condition: No tool_calls → routes to END      │
└──────────────────────────────────────────────────────┘
```

**Key Interview Point:** The LLM uses **function calling** (also called tool use). The tools are bound to the LLM via `llm.bind_tools(tools)`. The LLM doesn't execute the tools itself — it outputs a structured JSON describing which tool to call with what arguments. LangGraph's `ToolNode` then executes them.

---

## 8. RAG Architecture — Theory & Implementation

### What is RAG?

**Retrieval-Augmented Generation** = Instead of relying on the LLM's training data alone, we first RETRIEVE relevant documents from a knowledge base, then give them to the LLM as context.

### Our RAG Stack:

```
Document Loader:   PyPDFLoader (reads PDF → raw text)
Text Splitter:     RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
Embedding Model:   all-MiniLM-L6-v2 (HuggingFace, 384-dim, runs locally)
Vector Database:   FAISS (Meta AI, in-process, saved to disk)
Search Method:     Cosine/L2 similarity, top-3 results
Knowledge Base:    3 medical PDFs → 141 pages → 469 chunks
```

### Two Phases:

```
INGESTION (offline, run once via ingest.py):
  3 PDFs → 141 pages → 469 chunks → 469 embeddings → FAISS index on disk

RETRIEVAL (online, every chat):
  User query → embed → search FAISS → top-3 chunks → inject into LLM prompt
```

### Why These Specific Choices:

| Choice | Why |
|---|---|
| **all-MiniLM-L6-v2** | 80MB, runs locally, no API costs, 384-dim (fast search), trained on 1B+ sentence pairs |
| **FAISS** | In-process (no server), free, fast for small datasets, battle-tested by Meta |
| **chunk_size=1000** | ~200-250 tokens, focused enough for accurate retrieval |
| **chunk_overlap=200** | Prevents info loss at chunk boundaries |
| **top-3 results** | Enough context without overwhelming the LLM's context window |

---

## 9. OCR Architecture — Theory & Implementation

### Why Gemini Multimodal (not Tesseract)?

Traditional OCR = character-by-character pattern matching (fails on handwriting)
Gemini 1.5 Pro = Vision Transformer that **understands** the image holistically

### Flow:

```
Upload (JPEG/PNG) → PIL Image → Gemini 1.5 Pro API → Structured medical text
```

The extracted text serves **two purposes**:
1. **Enriches RAG query** — Appended to symptoms for more specific retrieval
2. **Injected into chatbot prompt** — LLM sees the full prescription text

---

## 10. Database & Auth Architecture

### Database Schema (SQLite):

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    Users      │     │  PatientProfiles  │     │  VisitHistory   │
├──────────────┤     ├──────────────────┤     ├─────────────────┤
│ id (PK)      │◄──┐ │ id (PK)          │◄──┐ │ id (PK)         │
│ email        │   └─│ user_id (FK)     │   └─│ patient_id (FK) │
│ hashed_pwd   │     │ name             │     │ timestamp       │
│ is_active    │     │ age              │     │ symptoms        │
└──────────────┘     │ gender           │     │ triage_label    │
                     │ chronic_conds    │     │ triage_confidence│
                     └──────────────────┘     │ ai_summary      │
                                              │ heart_rate      │
                     ┌──────────────────┐     │ systolic_bp     │
                     │  MRIPrediction   │     │ oxygen_sat      │
                     ├──────────────────┤     │ body_temp       │
                     │ id (PK)          │     │ pain_level      │
                     │ patient_id (FK)  │     │ lat, lon        │
                     │ diagnosis        │     │ location_string │
                     │ confidence       │     │ arrival_mode    │
                     │ triage_level     │     │ ocr_extracted   │
                     │ severity         │     └─────────────────┘
                     │ recommendation   │
                     └──────────────────┘
```

### Auth Flow:

```
Register:  password → bcrypt hash → stored in DB
Login:     password → bcrypt verify → JWT token (HS256, 7-day expiry)
Protected: Bearer token → decode JWT → extract email → load user from DB
```

---

## 11. API Architecture

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/auth/register` | POST | ❌ | Create account + profile |
| `/auth/login` | POST | ❌ | Get JWT token |
| `/chat/` | POST | ✅ | Main agent workflow (symptoms + vitals + image) |
| `/predict/` | POST | ❌ | ML triage prediction only |
| `/predict/mri` | POST | ✅ | Brain tumor classification |
| `/user/history` | GET | ✅ | Get patient profile + visit history |
| `/user/locator` | GET | ❌ | Standalone hospital/pharmacy search |

### The `/chat/` endpoint is the most complex:

```
POST /chat/
  Body (multipart/form-data):
    - symptoms: "chest pain, dizziness"
    - location: '{"lat": 19.07, "lon": 72.87}'
    - vitals_json: '{"age": 55, "heart_rate": 110, ...}'
    - file: prescription.jpg (optional)
    - thread_id: "abc-123" (optional, for conversation continuity)
  
  Processing:
    1. Parse vitals JSON → PatientVitals object
    2. Read image bytes if uploaded
    3. Reverse-geocode GPS → "Mumbai, Maharashtra"
    4. Load patient profile + visit history from DB
    5. Build initial AgentState
    6. Run LangGraph agent (all 5 nodes)
    7. Save visit to database
    8. Send email summary to patient
    9. Return: {thread_id, ml_prediction, ocr_extracted, agent_response}
```

---

## 12. Frontend Architecture

```
8 HTML Pages (vanilla HTML/CSS/JS, no framework):

index.html         → Landing page (hero, features)
login.html         → Login form → POST /auth/login
register.html      → Registration form → POST /auth/register
mode_select.html   → Choose: Manual Entry or Tumor Mode
manual_entry.html  → Main app: symptoms + vitals + image upload → POST /chat/
tumor_mode.html    → MRI upload → POST /predict/mri
dashboard.html     → Visit history timeline → GET /user/history
profile.html       → Patient profile view
locator.html       → Hospital/Pharmacy finder → GET /user/locator
```

---

## 13. Interview Q&A — 25 Questions They Will Ask

### Architecture & Design

**Q1: "Why did you choose LangGraph over a simple LangChain chain?"**
> "A simple chain is stateless — input goes in, output comes out. I needed a stateful agent that can: (1) run multiple preprocessing steps in sequence, (2) allow the LLM to autonomously decide to call tools, (3) loop back after tool execution, and (4) maintain conversation memory across API calls. LangGraph's StateGraph gives me all of that with a directed graph model."

**Q2: "What is the difference between a node and an edge in your graph?"**
> "A node is a processing function — it reads from the shared state, does work (like running an ML model or calling an API), and writes results back to the state. An edge defines the flow — which node runs after which. I also use conditional edges where the routing depends on the LLM's output — if it made tool calls, we go to the tools node; otherwise, we go to END."

**Q3: "How does conversation memory work?"**
> "LangGraph's MemorySaver checkpointer saves the entire AgentState (including the messages list) keyed by a thread_id. When the user sends a follow-up message with the same thread_id, the previous state is loaded, so the LLM sees the full conversation history. The messages field uses a reducer function that appends new messages instead of replacing them."

### RAG-Specific

**Q4: "Why RAG instead of fine-tuning?"**
> "Three reasons: (1) Medical guidelines change frequently — with RAG, I just add new PDFs without retraining. (2) RAG is transparent — I can see which documents were retrieved, which matters for medical AI accountability. (3) Fine-tuning a 70B model requires massive GPU resources and bakes knowledge into weights permanently."

**Q5: "What embedding model do you use and why?"**
> "all-MiniLM-L6-v2 — a distilled BERT model producing 384-dimensional vectors. I chose it because it runs entirely locally (80MB, no API costs), is fast (~14K sentences/sec), and was trained on 1B+ sentence pairs for strong semantic understanding. For a medical FYP, keeping data local matters."

**Q6: "What is FAISS and how does it search?"**
> "FAISS is Meta's library for vector similarity search. I use IndexFlatL2, which computes exact L2 (Euclidean) distance between the query vector and every stored vector. It returns the k-nearest vectors. With 469 chunks and 384 dimensions, this takes under 10ms."

**Q7: "What chunking strategy do you use?"**
> "RecursiveCharacterTextSplitter with 1000-character chunks and 200-character overlap. The recursive splitter tries to split at paragraph boundaries first, then sentences, then words — preserving natural text structure. The overlap ensures that if a drug interaction warning spans two chunks, both chunks contain the full warning."

### OCR-Specific

**Q8: "Why Gemini for OCR instead of Tesseract?"**
> "Medical prescriptions have handwritten text, abbreviations (Rx, BD, TDS), and poor image quality. Tesseract does character-level pattern matching and fails on these. Gemini 1.5 Pro is a vision-language model that understands the image holistically — it knows 'Rx' means prescription, 'BD' means twice daily, and can structure the output by medications and dosages."

**Q9: "Does the image go directly to the chatbot LLM?"**
> "No. The chatbot (Groq LLaMA-3) is text-only — it can't process images. The image first goes to Gemini 1.5 Pro (a multimodal model) which extracts text. That text string is then passed to the chatbot in its system prompt. So we use two different LLMs — Gemini for vision, LLaMA for reasoning."

### ML/DL Models

**Q10: "What ML model do you use for triage?"**
> "A Random Forest classifier trained on 9 features: age, heart rate, blood pressure, oxygen saturation, temperature, pain level, chronic disease count, ER visit history, and arrival mode. It predicts 4 triage levels (Non-Urgent to Emergent) with probability scores."

**Q11: "How does the brain tumor model work?"**
> "It's a CNN trained on brain MRI images, exported to ONNX format for fast CPU inference. It classifies into 4 categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor. Input is resized to 150×150, normalized to [0,1]. Each class maps to a triage level and clinical recommendation."

**Q12: "Why ONNX instead of TensorFlow/PyTorch directly?"**
> "ONNX is framework-agnostic and optimized for inference. It runs without needing the full TensorFlow/PyTorch installation, is faster at inference time, and is the standard for deploying models in production."

### Tools & APIs

**Q13: "How does the tool-calling mechanism work?"**
> "Tools are Python functions decorated with `@tool` and bound to the LLM via `llm.bind_tools()`. When the LLM decides it needs data (like nearby hospitals), it doesn't execute the tool — it outputs a structured JSON with the function name and arguments. LangGraph's ToolNode intercepts this, executes the function, and feeds the result back to the LLM as a ToolMessage."

**Q14: "What happens if the Google Places API fails?"**
> "I implemented a graceful degradation pattern. The hospital_locator first tries Google Places API. If that fails (rate limit, network error, missing API key), it automatically falls back to OpenStreetMap's Nominatim API, which is free and requires no key. This ensures the tool never crashes the entire agent."

### Auth & Security

**Q15: "How do you handle authentication?"**
> "JWT-based auth with bcrypt password hashing. On register, password is hashed with bcrypt (salt auto-generated). On login, password is verified against the hash, and a JWT token (HS256, 7-day expiry) is issued. Protected endpoints use FastAPI's Depends() with OAuth2PasswordBearer to extract and validate the token."

### System Design

**Q16: "How would you scale this to production?"**
> "Several changes: (1) Replace SQLite with PostgreSQL, (2) Replace MemorySaver with Redis-backed checkpointer, (3) Replace FAISS with Pinecone or Weaviate for distributed vector search, (4) Add Celery for async task processing, (5) Containerize with Docker, (6) Add rate limiting and input validation."

**Q17: "What are the limitations of your system?"**
> "Honest answer: (1) MemorySaver is in-memory, so conversations are lost on server restart. (2) FAISS doesn't scale to millions of documents. (3) The email tool is synchronous and blocks the response. (4) No streaming — the user waits for the full pipeline to finish. (5) No authentication on the ML prediction endpoint."

**Q18: "How do you handle errors in the agent pipeline?"**
> "Every node has try-except blocks that return graceful fallbacks instead of crashing. If Gemini fails, OCR returns 'Error extracting text'. If FAISS is empty, retrieval returns empty context. If the LLM fails, a SystemMessage with the error is returned. The /chat/ endpoint wraps everything in a try-except with db.rollback() on failure."

### Conceptual / Theory

**Q19: "What is the difference between an agent and a chain?"**
> "A chain is a fixed sequence: A → B → C → done. An agent has autonomy — it can decide its next action based on the current state. My chatbot node acts as the agent brain: it looks at the context and decides whether to call tools or give a final answer. The key difference is the conditional edge and the loop."

**Q20: "What is semantic search vs keyword search?"**
> "Keyword search (TF-IDF, BM25) matches exact words — 'heart attack' won't find 'myocardial infarction'. Semantic search embeds both query and documents into vector space where similar meanings are close together. So 'heart attack' and 'myocardial infarction' end up near each other because the embedding model learned they mean the same thing."

**Q21: "What is a vector embedding?"**
> "A dense numerical representation (list of numbers) that captures the semantic meaning of text. My model produces 384-dimensional vectors. Similar texts produce similar vectors, and we measure similarity using distance metrics like L2 or cosine."

**Q22: "Explain the RAG formula conceptually."**
> "P(answer | question) = sum over retrieved documents of P(doc is relevant) × P(answer given doc + question). Instead of the LLM generating from memory alone, it first retrieves relevant documents, then generates using those documents as additional context."

**Q23: "What is function calling in LLMs?"**
> "Function calling is when an LLM outputs structured tool-call JSON instead of natural language. The LLM doesn't execute the function — it describes which function to call and with what arguments. The application layer (LangGraph in my case) then executes the function and feeds the result back to the LLM."

**Q24: "What is a checkpointer in LangGraph?"**
> "A checkpointer serializes and saves the entire graph state (all variables in AgentState) after each node execution. When a new request comes with the same thread_id, it loads the saved state, enabling multi-turn conversations. MemorySaver stores in RAM (dev), but production would use Redis or PostgreSQL."

**Q25: "If you had one more month, what would you add?"**
> "Three things: (1) Per-patient RAG — embed each visit summary into a personal vector store so the agent can semantically search a patient's full medical history, not just the last 3 visits. (2) Drug interaction RAG — a dedicated knowledge base for flagging dangerous drug combinations from OCR-extracted prescriptions. (3) Streaming responses using Server-Sent Events so the user sees the response being generated in real-time."

---

> **Good luck with the interview tomorrow! You've built a genuinely impressive multi-AI system. Own it.** 🚀

