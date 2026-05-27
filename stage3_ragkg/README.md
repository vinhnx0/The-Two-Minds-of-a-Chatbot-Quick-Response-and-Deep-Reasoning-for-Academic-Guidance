# Stage 3 — IUStudyGuide RAG + Knowledge Graph + Curriculum Planner

## Overview

`stage3_ragkg` is the third stage of a thesis project that builds an academic study guide and curriculum question-answering system for IU students.

The system combines:

- Retrieval-Augmented Generation (RAG)
- Curriculum Knowledge Graph reasoning
- FAST/DEEP query routing
- LLM-based question decomposition and answer synthesis
- ILP/CP-SAT curriculum planning
- Streamlit demo interface

At a high level, the system works with two reasoning modes:

- **FAST Route — LLM-based engine**: handles simpler academic questions such as course lookup, credits, offered semesters, and prerequisite explanations. It uses retrieved evidence, KG findings, and an LLM to generate quick grounded answers.
- **DEEP Route — Algorithm-based engine**: handles complex curriculum planning questions that require deeper constraint reasoning. It uses an ILP/CP-SAT optimization algorithm to generate feasible multi-semester study plans.

In simple terms, the system helps students ask questions about courses, prerequisites, credit requirements, and possible study plans.

---

## Why this project matters

Academic curriculum information is often spread across PDFs, official program documents, course lists, and prerequisite rules. This makes it difficult for students to quickly answer questions such as:

- “What are the prerequisites for this course?”
- “Can I take this course after completing these subjects?”
- “How should I plan my remaining semesters?”
- “Which courses should come before Machine Learning?”

This project explores how an AI system can combine unstructured documents with structured curriculum knowledge to provide more grounded and explainable academic guidance.

---

## Key Features

### Hybrid Retrieval

- Loads academic PDFs, text, and markdown documents.
- Splits documents into chunks.
- Embeds chunks using SentenceTransformers.
- Stores dense vectors in Qdrant.
- Builds a BM25 sparse index.
- Combines BM25 and dense retrieval.
- Applies MMR to improve result diversity.

### Course Alias Normalization

- Resolves course codes and course-like aliases.
- Supports exact, fuzzy, and semantic fallback matching.
- Helps map user questions to canonical course IDs in the Knowledge Graph.

### Curriculum Knowledge Graph

- Loads curriculum data from `knowledge_graph.json`.
- Validates course nodes, credits, semesters, and prerequisite edges.
- Builds incoming and outgoing prerequisite maps.
- Produces compact KG evidence for LLM grounding.

### Prerequisite Reasoning

- Extracts prerequisite chains for a target course.
- Converts prerequisite chains into topological ordering.
- Produces user-facing prerequisite findings such as:
  - required courses
  - dependency rules
  - prerequisite levels
  - course display names

### FAST / DEEP Routing

The system routes each query into one of two paths:

- **FAST route**: for course lookup, prerequisite reasoning, and general academic Q&A.
- **DEEP route**: for multi-semester curriculum planning.

Only curriculum planning questions are routed to the DEEP planner.

### LLM-Based Planning Support

The system uses local LLM calls for:

- question decomposition
- structured JSON extraction
- routing override
- answer synthesis
- planning constraint extraction

### ILP / CP-SAT Curriculum Planner

The DEEP planner uses OR-Tools CP-SAT to generate feasible study plans under academic constraints, including:

- semester availability
- prerequisite ordering
- completed courses
- avoided courses
- credit limits per semester
- minimum credit load rules
- compulsory courses
- elective credit requirements
- thesis or thesis-replacement track constraints

### Streamlit Demo UI

The Streamlit app connects the full pipeline into an interactive demo with:

- question input
- route decision display
- retrieval evidence
- KG evidence
- generated answers
- curriculum planning mode
- index build/rebuild controls

### Guardrails and Observability

- Input guardrails remove simple prompt-injection phrases and redact basic PII.
- Output guardrails can enforce citation checks.
- Logging tracks routing decisions, KG evidence size, retrieval status, LLM calls, and request runtime.

---

## System Architecture

```text
User Question
    |
    v
Input Guardrails
    |
    v
Course Entity Detection + Alias Normalization
    |
    v
Question Decomposition
    |
    v
Hybrid Retrieval
(BM25 + Dense Qdrant + MMR)
    |
    v
Router
    |
    +-----------------------------+
    |                             |
(LLM-based engine)      (Algorithm-based engine)
    |                             |
    v                             v
FAST Pipeline                  DEEP Pipeline
Course lookup /                Curriculum planning
Prerequisite reasoning         with ILP / CP-SAT
General Q&A
    |                             |
    v                             v
KG Findings                   KG Findings
RAG Evidence                  Curriculum Plan
    |                             |
    v                             v
Answer Synthesis              Planning Answer Synthesis
    |                             |
    +-------------+---------------+
                  |
                  v
            Streamlit UI
````

---

## Two Reasoning Modes: FAST LLM Engine vs DEEP Algorithm Engine

### FAST Pipeline

Used for most questions, including:

* course details
* course credits
* offered semester
* prerequisite questions
* eligibility-style questions
* general academic questions

FAST pipeline flow:

```text
Question
→ sanitize input
→ detect course entities
→ retrieve relevant chunks
→ build compact KG findings
→ select answer template
→ synthesize grounded answer
```

For prerequisite questions, the system can also render a deterministic prerequisite answer directly from KG findings instead of relying fully on LLM generation.

### DEEP Pipeline

Used only when the user asks for a multi-semester curriculum plan.

DEEP pipeline flow:

```text
Planning question
→ extract constraints
→ detect completed courses / semester / credit limits
→ build curriculum graph
→ formulate CP-SAT model
→ solve feasible schedule
→ synthesize planning explanation
```

The DEEP route is designed for heavier reasoning because it must satisfy multiple academic constraints at once.

---

## Tech Stack

* **Python**
* **Streamlit** — demo UI
* **Qdrant** — dense vector store
* **rank-bm25** — sparse retrieval
* **SentenceTransformers** — embedding model
* **OR-Tools CP-SAT** — ILP-style curriculum planning
* **Pydantic** — schema validation
* **pypdf** — PDF text extraction
* **RapidFuzz** — fuzzy alias matching
* **Local OpenAI-compatible LLM backend** — to be confirmed based on local setup

---

## Folder Structure

```text
stage3_ragkg
├─ app
│  ├─ guardrails.py
│  ├─ kg
│  │  ├─ loader.py
│  │  └─ reasoning.py
│  ├─ llm.py
│  ├─ logging_utils.py
│  ├─ planner.py
│  ├─ prompts
│  │  ├─ answer_synth_prompt.txt
│  │  ├─ planner_prompt.txt
│  │  ├─ queries_prompt.txt
│  │  ├─ router_prompt.txt
│  │  └─ slow_planning_synth_prompt.txt
│  ├─ rag
│  │  ├─ alias_normalizer.py
│  │  ├─ bm25.py
│  │  ├─ chunker.py
│  │  ├─ embedder.py
│  │  ├─ hybrid.py
│  │  ├─ indexing.py
│  │  └─ utils_text.py
│  ├─ router.py
│  ├─ slow_curriculum_ilp.py
│  └─ streamlit_app.py
├─ config.yaml
└─ pyproject.toml
```

---

## How to Run

### 1. Install dependencies

```bash
cd stage3_ragkg
pip install -e .
```

Or install using the dependency manager defined in `pyproject.toml`.

### 2. Prepare data

Make sure the parent `data/` folder contains:

```text
data/
├─ kg/
│  └─ knowledge_graph.json
├─ CTDT-K23-Nganh-Khoa-hoc-Du-lieu-trang-33-Signed-4.pdf
├─ Ke-hoach-tuyen-sinh-2025.pdf
├─ NGÀNH KHOA HỌC DỮ LIỆU.pdf
├─ QD_Vv-ban-hanh-Quy-che-2025-0512-Signed.pdf
└─ Thong-tin-tuyen-sinh-2025.pdf
```

### 3. Configure the system

Update `config.yaml` if needed:

* data path
* KG JSON path
* Qdrant collection name
* BM25 index path
* embedding model
* LLM backend
* logging settings

### 4. Start Qdrant

If using Qdrant server mode:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

If using embedded mode, Qdrant will store local data based on the configured path.

### 5. Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

### 6. Build or rebuild the index

If the index is missing, the Streamlit UI will show an option to build or rebuild the index.

The indexing process will:

```text
load documents
→ clean text
→ chunk documents
→ enrich chunks with course aliases
→ save chunks JSONL
→ build BM25
→ embed chunks
→ store vectors in Qdrant
```

---

## Example Questions

### Course lookup

```text
Môn IT140IU được mở vào học kỳ nào và bao nhiêu tín chỉ?
```

### Prerequisite reasoning

```text
Môn IT172IU có yêu cầu tiên quyết là những môn nào?
```

### Eligibility-style question

```text
Tôi đã học MA001IU và IT149IU thì học được IT154IU chưa?
```

### Curriculum planning

```text
Mình đã học xong HK1 với các môn MA001IU, MA026IU, EN007IU, EN008IU, IT135IU, PT001IU, PE015IU, PE016IU. Hãy sắp xếp các môn còn lại từ HK2 đến HK8 để tốt nghiệp.
```

---

## Current Limitations

* The system depends on the quality and completeness of `knowledge_graph.json`.
* PDF parsing currently uses basic text extraction, so complex layouts may not be perfectly captured.
* The BM25 tokenizer is simple whitespace tokenization.
* LLM behavior depends on the local model and configuration.
* Curriculum planning is constrained by the rules currently encoded in the KG and CP-SAT model.
* Some academic policies may require manual confirmation against official IU documents.
* Hypothetical question generation for chunks is currently present as a placeholder.
* Production authentication, deployment, and user management are not included.

---

## Future Improvements

* Improve PDF parsing for tables and complex curriculum layouts.
* Add evaluation metrics for retrieval, prerequisite reasoning, and planning accuracy.
* Add more robust Vietnamese text normalization and tokenization.
* Expand the Knowledge Graph with more academic rules and policy metadata.
* Add a clearer admin workflow for updating curriculum data.
* Add automated tests for KG validation, routing, retrieval, and ILP constraints.
* Improve UI explanations for infeasible plans.
* Add export options for generated study plans.
* Deploy the system as a web service with persistent storage and monitoring.

---

## Portfolio Summary

This stage demonstrates an AI academic advising system that combines RAG, structured Knowledge Graph reasoning, query routing, and constraint-based optimization.

It is designed as a thesis and portfolio project showing practical skills in:

* GenAI system design
* retrieval engineering
* Knowledge Graph grounding
* LLM orchestration
* optimization-based planning
* AI application development
* explainable academic decision support
