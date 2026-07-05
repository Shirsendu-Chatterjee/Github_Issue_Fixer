# GitHub Issue Fixer

## Table of Contents

* Project Summary
* Architecture
* Features
* Processing Pipeline
* Component Breakdown
* Retrieval & Code Understanding
* Technology Stack
* Current Limitations
* Future Roadmap
* Contact

---

# Project Summary

**GitHub Issue Fixer** is an AI-powered GitHub issue triage and analysis system that combines **code-aware Retrieval-Augmented Generation (RAG)**, **Abstract Syntax Tree (AST) analysis**, and **agentic retrieval planning** to automatically understand repository codebases and generate technical responses for GitHub issues.

Unlike traditional document RAG systems that simply split files into fixed-size chunks, this project performs **structure-aware code parsing**, generates semantic summaries for code chunks using an LLM, retrieves repository-specific context through vector search, and iteratively plans additional retrieval steps before generating a final response.

The entire workflow is orchestrated using **LangGraph**, allowing every stage of reasoning to be represented as a modular state machine.

---

# Architecture

```
GitHub Repository
        │
        ▼
Repository Crawling
        │
        ▼
AST-based Code Chunking
        │
        ▼
LLM Code Summarization
        │
        ▼
Code-aware Embeddings
(st-codesearch-distilroberta-base)
        │
        ▼
FAISS Vector Index
        │
        ▼
GitHub Issue
        │
        ▼
Issue Classification
        │
        ▼
Agentic Retrieval Loop
(Retrieve → Evaluate → Retrieve Again)
        │
        ▼
Optional Wikipedia Retrieval
        │
        ▼
Specialist LLM
(Bug / Feature / Documentation / Security)
        │
        ▼
Markdown Synthesis
        │
        ▼
Automatic GitHub Comment
```

---

# Features

* AST-based semantic chunking for Python source code
* Function-aware and class-aware repository indexing
* JavaScript/TypeScript structural chunking
* Automatic LLM-generated summaries for every code chunk
* Code-search-optimized embedding model
* FAISS semantic retrieval
* Multi-hop agentic retrieval planning
* Intelligent Wikipedia usage only when external knowledge is beneficial
* Issue classification into:

  * Bug
  * Feature
  * Documentation
  * Security
  * Unknown
* Specialist prompts tailored to each issue category
* Automatic Markdown formatting
* Automatic GitHub issue commenting
* LangGraph orchestration for modular execution

---

# Processing Pipeline

## 1. Repository Ingestion

The repository is traversed using **PyGithub**.

Supported file types include:

* Python
* JavaScript
* TypeScript
* Markdown
* JSON
* YAML
* HTML
* CSS
* Shell
* Text
* Jupyter notebooks

Large binary files are ignored to reduce unnecessary processing.

---

## 2. Structure-Aware Code Chunking

Instead of splitting code into arbitrary fixed-size windows, the project parses source code structurally.

### Python

Python files are parsed using the built-in **AST module**.

Chunks are created for:

* top-level functions
* class methods
* class bodies
* module-level code

This preserves logical program boundaries and avoids splitting functions across multiple chunks.

### JavaScript / TypeScript

A lightweight parser identifies:

* functions
* arrow functions
* classes

using regular-expression boundary detection.

### Other Files

Non-code files fall back to fixed-size chunking.

---

## 3. LLM Code Summaries

Every code chunk is summarized into a concise natural-language description.

Example:

```
Summary:
Authenticates users using OAuth2 and returns an access token.
```

These summaries improve retrieval by allowing natural-language GitHub issues to match implementation semantics rather than only code tokens.

---

## 4. Code Embeddings

Repository chunks are embedded using:

**st-codesearch-distilroberta-base**

This model is trained specifically for mapping natural-language descriptions to source code, making it significantly better suited for code retrieval than general-purpose sentence embedding models.

---

## 5. Semantic Retrieval

Embeddings are indexed using **FAISS IndexFlatL2**.

When a GitHub issue is processed, semantic similarity search retrieves the most relevant repository context before reasoning begins.

---

## 6. Agentic Retrieval Planning

Rather than retrieving repository context only once, the system performs an iterative retrieval loop.

The LLM evaluates whether sufficient repository evidence has been gathered.

If additional context is needed, it generates a refined search query and performs another retrieval cycle.

This process continues until:

* enough evidence has been collected, or
* the configured retrieval limit is reached.

---

## 7. Intelligent Wikipedia Retrieval

External knowledge is not always required.

A planning node first determines whether background information from Wikipedia would improve the final response.

Examples include:

* protocols
* algorithms
* standards
* security concepts

If unnecessary, this step is skipped entirely.

---

## 8. Specialist Reasoning

Different issue categories invoke different reasoning templates.

Supported specialists include:

* Bug Analysis
* Documentation Review
* Feature Planning
* Security Assessment

Each specialist receives:

* GitHub issue content
* retrieved repository context
* optional Wikipedia context

before generating a technical response.

---

## 9. Markdown Synthesis

The generated response is cleaned and formatted into readable Markdown suitable for GitHub discussions.

---

## 10. Automatic GitHub Commenting

The final response is automatically posted to the corresponding GitHub issue using the GitHub API.

---

# Component Breakdown

| Component          | Responsibility                                         |
| ------------------ | ------------------------------------------------------ |
| Repository Fetcher | Downloads repository files                             |
| AST Chunker        | Generates semantic code chunks                         |
| Summary Generator  | Produces natural-language summaries                    |
| Embedding Model    | Converts code into dense vectors                       |
| FAISS              | Performs semantic nearest-neighbor retrieval           |
| Retrieval Planner  | Determines whether more repository context is required |
| Wiki Planner       | Decides if external knowledge is useful                |
| Specialist         | Produces issue-specific technical recommendations      |
| Synthesizer        | Formats clean Markdown                                 |
| Commenter          | Posts results back to GitHub                           |

---

# Retrieval & Code Understanding

This project improves code understanding through multiple complementary techniques:

* AST-aware semantic chunking
* Code-search-specific embeddings
* Natural-language code summaries
* Repository-aware retrieval
* Agentic multi-hop search
* Optional external knowledge grounding

Together these components enable more accurate retrieval than traditional fixed-window document chunking.

---

# Technology Stack

* Python
* LangGraph
* LangChain
* ChatGroq
* PyGithub
* SentenceTransformers
* FAISS
* NumPy
* Wikipedia API
* Python AST

---

# Current Limitations

* FAISS index is built entirely in memory.
* Repository summaries are regenerated each startup rather than cached.
* Retrieval currently relies on dense vector similarity without reranking.
* Cross-file call graphs are not yet incorporated.
* Repository maps are not yet generated.
* Index updates are not incremental.

---

# Future Roadmap

* Persistent vector databases (Qdrant or LanceDB)
* Hybrid retrieval (Dense + BM25)
* Cross-encoder reranking
* AST-derived call graph construction
* Repository map generation
* Cached code summaries
* Incremental repository indexing
* GitHub App deployment
* CI/CD integration
* Multi-model routing
* Repository knowledge graph
* Automated evaluation benchmarks

---

# Contact

Contributions, suggestions, and issue reports are welcome. Feel free to open an issue or submit a pull request to help improve the project.
