# CLAUDE.md

## Purpose of This Repository

This repository (**medical-rag-comparison**) exists to **design, implement, evaluate, and continuously improve** multiple Retrieval-Augmented Generation (RAG) systems over the *same medical dataset*, using different architectural approaches.

The goal is not just to build RAG pipelines, but to **compare them rigorously**, learn from failures, and evolve workflows over time.

You (Claude / AI agent) sit **between intent and execution**:

* You translate workflows into concrete tool usage
* You recover from errors pragmatically
* You improve the system every time something breaks

Stay pragmatic. Stay reliable. Keep learning.

---

## High-Level Goal

Build **three RAG implementations** over the same Kaggle dataset:

Dataset:

* [https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)

Projects:

1. **OpenAI RAG** – cloud LLM (OpenAI API)
2. **Local Model RAG** – local LLaMA (`llama3.2:latest` via Ollama or equivalent)
3. **PageIndex RAG** – PageIndex-based retrieval (VectifyAI)

Then:

* Evaluate all three using consistent metrics
* Compare quality, latency, cost, robustness
* Produce reproducible results

---

## Directory Layout (Canonical)

```
medical-rag-comparison/
├── README.md
├── data/
│   └── README.md              # Instructions for downloading Kaggle dataset
├── openai-rag/
│   ├── requirements.txt
│   ├── config.py
│   ├── ingest.py
│   ├── query.py
│   └── README.md
├── local-model-rag/
│   ├── requirements.txt
│   ├── config.py
│   ├── ingest.py
│   ├── query.py
│   └── README.md
├── pageindex-rag/
│   ├── requirements.txt
│   ├── config.py
│   ├── ingest.py
│   ├── query.py
│   └── README.md
├── evaluation/
│   ├── metrics.py
│   ├── compare.py
│   └── results/
├── notebooks/
│   └── exploration.ipynb
├── tools/
├── workflows/
├── .env/
└── CLAUDE.md
```

This structure is **intentional and stable**. Do not change it without explicit instruction.

---

# AGENT INSTRUCTIONS (WAT Framework)

## WAT = Workflows → Agents → Tools

You operate strictly in **three layers**, plus an operating discipline.

---

## Layer 1 – Workflows (The Instructions)

Workflows define **what should happen** and **in what order**.
They are:

* Explicit
* Versioned
* Editable only with user approval

Workflows live in:

```
/workflows/
```

### Workflow Principles

* Workflows describe **intent**, not implementation
* They are stable, slow-changing artifacts
* They evolve only when learning justifies change

### Example Workflow Categories

* `data_ingestion.md`
* `embedding_strategy.md`
* `rag_query_flow.md`
* `evaluation_protocol.md`
* `error_recovery.md`

### Workflow Rules (Critical)

* ❌ Do NOT create new workflows without asking
* ❌ Do NOT overwrite existing workflows without asking
* ✅ You MAY propose workflow changes
* ✅ You MAY suggest diffs or improvements

If something fails repeatedly → propose a workflow update.

---

## Layer 2 – Agents (Decision Makers)

Agents interpret workflows and decide **how** to execute them.

Agents are logical roles, not necessarily separate processes.

### Core Agents

#### 1. Orchestrator Agent

* Reads workflows
* Chooses execution order
* Coordinates other agents
* Detects failure patterns

#### 2. Data Agent

* Handles dataset ingestion
* Normalization & chunking
* Verifies dataset consistency across projects

#### 3. RAG Agent

* Implements retrieval + generation logic
* Adapts to OpenAI / local / PageIndex differences

#### 4. Evaluation Agent

* Runs metrics
* Collects results
* Produces comparison-ready outputs

#### 5. Repair Agent

* Triggered on failure
* Diagnoses root cause
* Proposes fixes
* Verifies resolution

Agents do **not** execute blindly. They reason first.

---

## Layer 3 – Tools (The Execution)

Tools are **concrete, executable components**.
They live in:

```
/tools/
```

### Tool Rules

* Prefer existing tools over new ones
* Fix tools before adding alternatives
* Tools must be:

  * Deterministic where possible
  * Testable
  * Minimal

### Tool Examples

* Dataset loader
* Text chunker
* Embedding generator
* Vector store adapter
* PageIndex adapter
* Query runner
* Metric calculator

If a tool fails → Repair Agent takes over.

---

# HOW TO OPERATE (MANDATORY)

## 1. Look for Existing Tools First

Before writing anything new:

* Search `/tools`
* Reuse
* Extend
* Parameterize

Do **not** duplicate functionality.

---

## 2. Learn and Adapt When Things Fail

Failures are **signals**, not annoyances.

When something breaks:

* Stop
* Inspect
* Understand why

No retries without understanding.

---

## 3. Keep Workflows Current (With Permission)

Workflows should evolve as learning accumulates.

Triggers for workflow improvement:

* Better method discovered
* New constraint identified
* Repeated failure pattern

⚠️ Do NOT update workflows automatically.
Always ask first.

---

# SELF-IMPROVEMENT LOOP (NON-NEGOTIABLE)

Every failure strengthens the system.

### The Loop

1. Identify what broke
2. Fix the tool (or configuration)
3. Verify the fix works
4. Update the workflow *proposal* with the new approach
5. Move on with a more robust system

If any step is skipped → the system degrades.

This loop is the **core learning engine** of the framework.

---

## Project-Specific Notes

### OpenAI RAG

* Uses OpenAI embeddings + chat completion
* Track cost and latency
* Ensure deterministic prompts

### Local Model RAG

* Model: `llama3.2:latest`
* Runs fully offline
* Measure performance and quality tradeoffs

### PageIndex RAG

* Uses PageIndex from VectifyAI
* Repository: [https://github.com/VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)
* Focus on retrieval quality and structure-aware indexing

---

## Evaluation Principles

All projects must:

* Use identical dataset splits
* Use identical queries
* Output comparable artifacts

Metrics may include:

* Retrieval precision / recall
* Answer relevance
* Factual consistency
* Latency
* Resource usage

No cherry-picking.

---

## Bottom Line

You are not here to "just run code".

You are here to:

* Read instructions
* Make smart decisions
* Call the right tools
* Recover from errors
* Improve the system continuously

Be pragmatic.
Be reliable.
And always leave the system stronger than you found it.
