# github-issue-fixer

![Project: Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![Project: Research/Prototype](https://img.shields.io/badge/status-prototype-orange)

Table of contents
- Project summary
- Implementation overview
- Component breakdown
- Data flow & algorithms
- Key libraries and rationale
- Limitations & tests
- Future roadmap & integration ideas
- Contact

Project summary
----------------
github-issue-fixer is a Python prototype that automates technical issue triage and initial remediation suggestions for GitHub repositories by combining retrieval-augmented generation (RAG) with lightweight programmatic orchestration. The tool scans a repository, builds a vector index of repository content, and runs a state-machine pipeline that classifies open issues, retrieves context (repo + wiki), invokes an LLM to produce targeted specialist output, and formats the result for posting back to GitHub.

This repository contains two primary scripts used during development: `new.py` (primary implementation) and `main.py` (alternate/older flow). The implementation targets modern transformer embeddings and FAISS for fast similarity search, and uses a small state-graph orchestrator to structure the reasoning workflow.

Implementation overview
-----------------------
- Purpose: enable automated, context-aware issue analysis by combining local repository signals (file contents, README, code) with external knowledge (Wikipedia) and an LLM-driven reasoning pipeline.
- Design goals: modular nodes for discrete responsibilities (classification, retrieval, specialist reasoning, synthesis, and commenting), reproducible prompt patterns, and a retrieval layer to ground LLM output in repository context.
- Execution model: synchronous Python runtime that builds an in-memory FAISS index at startup and then iterates over open GitHub issues, invoking the compiled `StateGraph` for each issue.

Component breakdown
-------------------
- `new.py` (primary)
  - Repo ingestion: `_fetch_repo_files(max_files)` walks repository contents via PyGithub and collects text from common file types.
  - Chunking/embedding: `_chunk()` splits large files and `SentenceTransformer` (`all-MiniLM-L6-v2`) encodes chunks into embeddings.
  - Indexing: `build_faiss_index()` constructs a `faiss.IndexFlatL2` index and stores encoded chunks for RAG lookup.
  - Retrieval helpers: `rag_search(query, k)` performs nearest-neighbour search over the FAISS index; `wiki_search()` fetches succinct summaries from Wikipedia as an auxiliary context.
  - State nodes (LangGraph):
    - `classifier_node`: prompts an LLM to categorize an issue and extract a short wiki keyword.
    - `rag_node`: collects repository-grounded context for the issue.
    - `wiki_node`: retrieves external background knowledge.
    - `specialist_node`: issues a role-specific prompt (bug/feature/docs/security) to the LLM to produce remediation or recommendations.
    - `synthesizer_node`: formats the LLM output into clean markdown.
    - `commenter_node`: posts the synthesized result back to GitHub via PyGithub.
  - Runtime: `run()` enumerates open issues and runs the compiled StateGraph for each.

- `main.py` (auxiliary)
  - Contains an earlier state-machine variant with simpler branching between `doc_solver` and `bug_fixer` nodes and demonstrates a straightforward prompt/response loop.

Data flow & algorithms
----------------------
- Repository ingestion → chunking → embedding → FAISS index construction
  - Chunking uses fixed-size windows (CHUNK_SIZE=400) to limit each vector payload.
  - Embeddings: `SentenceTransformer(all-MiniLM-L6-v2)` produces dense vectors optimized for semantic similarity at small scale.
  - Index: `faiss.IndexFlatL2` is used for approximate nearest neighbour search (flat L2 index; fast and suitable for moderate-scale indexes).
- Issue processing pipeline
  - Input: GitHub issue (title, body, labels).
  - Classification: LLM assigns an issue type (bug/documentation/feature/security/unknown) and suggests a short wiki query.
  - Retrieval: RAG search on local repo chunks and Wikipedia summary provide grounding/context.
  - Specialist reasoning: LLM uses structured system prompts tuned per issue type to produce root cause, debugging steps, or implementation guidance.
  - Synthesis: LLM is asked to produce clean markdown for human consumption; optionally posted back to GitHub.

Key libraries and rationale
--------------------------
- PyGithub (`pygithub`): programmatic access to repository files, issues, labels, and issue comments.
- SentenceTransformers: compact sentence embeddings for semantic retrieval; offers strong performance with small footprint.
- FAISS (`faiss-cpu`): highly optimized vector index for nearest-neighbour search; `IndexFlatL2` chosen for simplicity and deterministic nearest-neighbour results.
- LangChain-Groq / ChatGroq: LLM invocation wrapper used to call a Groq-backed large model with temperature control and prompt lifecycle.
- LangGraph: lightweight state-graph orchestrator to express the processing pipeline as nodes and edges; aids in modularity and testing.
- Wikipedia: quick external knowledge summaries to augment repository context when issues require domain background.

Limitations & tests
-------------------
- Hard-coded credentials and API keys appear in `new.py`/`main.py`. This is unsafe; any production usage must replace inline secrets with secure environment management (e.g., secrets manager, environment variables, or GitHub Actions secrets).
- The current indexing is in-memory and synchronous; building the FAISS index at startup loads entire embeddings into RAM and blocks the process until complete. For larger repositories, use on-disk vector stores or incremental indexing.
- Error handling is minimal — many network calls (GitHub, Wikipedia, LLM API) are wrapped with broad excepts that swallow failures. Tests should assert behaviour on API failures and rate-limits.
- No automated tests are present. Recommended minimal tests:
  - Unit tests for chunking and embedding pipeline (deterministic chunk sizes, handling of binary files).
  - Integration test that runs a mock `StateGraph` against a small local repo snapshot and asserts node outputs for a representative issue.
  - Mocked API tests for GitHub and LLM interactions (use VCR or responses for HTTP mocking).

Future roadmap & integration ideas (technical)
--------------------------------------------
1. Robust secret management and CI integration
   - Remove inline secrets; support env-based configuration and a clear `.env.example` for development.
   - Add GitHub Actions workflows for linting, unit tests, and static analysis.

2. Modularize and package the core pipeline
   - Factor ingestion, retrieval, LLM orchestration, and GitHub I/O into separate modules with explicit interfaces.
   - Publish as a lightweight package with entrypoints for programmatic usage and integration into CI pipelines.

3. Scalable retrieval
   - Move from in-memory `IndexFlatL2` to a persistent, sharded vector store (e.g., FAISS on-disk, Milvus, Weaviate) with incremental updates.
   - Add chunk metadata (file path, line ranges) and passage scoring to produce actionable citations in LLM output.

4. Deterministic prompts and evaluation
   - Formalize prompt templates and add automated evaluation harnesses (BLEU/ROUGE-like where applicable, or human-in-the-loop scoring) to validate that suggested fixes are accurate and actionable.
   - Add shadow runs for pull requests and measure false-positive/false-negative triage rates.

5. Resilience and monitoring
   - Switch to async I/O (async GitHub client, async LLM calls) to improve throughput.
   - Instrument with telemetry: request latencies, LLM token consumption, success/failure rates, number of issues processed.

6. Multi-model fallback and cost control
   - Add a multi-LLM strategy: cheap model for classification, higher-quality model for synthesis, and a deterministic fallback for critical security findings.
   - Add token budgeting and throttling to control costs.

7. Integration surfaces
   - GitHub App / Bot: run as an app to react to issue events (open/label change) instead of polling.
   - CI Hooks: run the triage pipeline in pull requests to auto-generate diagnostics or documentation suggestions.
   - Slack/MS Teams / Email notifications: configurable sinks for triage summaries.

Contact
-------
For technical questions about architecture or to request specific resume-friendly summaries, inspect `new.py` for implementation details or open an issue in this repository.
