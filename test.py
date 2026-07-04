import os, re, ast
from typing import TypedDict, Literal, Optional
from langchain_groq import ChatGroq
import github as gh_lib
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
import wikipedia as wiki_lib
from langgraph.graph import StateGraph, START, END

os.environ["GROQ_API_KEY"] = ""
GITHUB_TOKEN = ""
REPO_NAME    = "Shirsendu-Chatterjee/ei"

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.2)

auth = gh_lib.Auth.Token(GITHUB_TOKEN)
g    = gh_lib.Github(auth=auth)
repo = g.get_repo(REPO_NAME)

wiki_lib.set_lang("en")

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CHUNK_SIZE      = 400   # fallback char-chunk size for non-code files
MAX_CHUNK_CHARS = 1200  # any single function/class chunk larger than this gets sub-split
_rag_chunks: list[str] = []


# --------------------------------------------------------------------------
# 1. Repo fetching — now returns (path, raw_text) so chunking can be
#    dispatched per-file based on extension.
# --------------------------------------------------------------------------

def _fetch_repo_files(max_files: int = 60) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    try:
        contents = repo.get_contents("")
        queue = list(contents)
        visited = 0
        while queue and visited < max_files:
            item = queue.pop(0)
            if item.type == "dir":
                queue.extend(repo.get_contents(item.path))
            elif item.type == "file" and item.size < 80_000:
                ext = item.path.rsplit(".", 1)[-1].lower()
                if ext in {"py", "md", "txt", "js", "ts", "json", "yaml", "yml", "html", "css", "rst", "ipynb", "sh"}:
                    try:
                        raw = item.decoded_content.decode("utf-8", errors="ignore")
                        files.append((item.path, raw))
                        visited += 1
                    except Exception:
                        pass
    except Exception:
        pass
    return files


# --------------------------------------------------------------------------
# 2. Function/class-aware chunking.
#    - .py files: real AST parsing -> one chunk per top-level function,
#      one chunk per class method, plus module-level leftovers.
#    - .js/.ts files: regex-based function/class boundary detection
#      (no full JS parser dependency, but structurally aware).
#    - everything else: falls back to fixed-size character chunking.
#    Any chunk that's still too big gets sub-split by size as a safety net.
# --------------------------------------------------------------------------

def _node_span(node: ast.AST) -> tuple[int, int]:
    start = node.lineno
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        start = min(start, decorators[0].lineno)
    end = getattr(node, "end_lineno", node.lineno)
    return start, end


def _chunk_python(text: str) -> list[tuple[str, str]]:
    """AST-based chunking: one chunk per top-level function, one per class
    method, module-level code (imports, constants, etc.) grouped separately."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [("module", text)]

    lines = text.splitlines()
    top_nodes = sorted(ast.iter_child_nodes(tree), key=lambda n: getattr(n, "lineno", 0))
    if not top_nodes:
        return [("module", text)]

    chunks: list[tuple[str, str]] = []
    prev_end = 0

    def _flush_gap(start_line: int, label: str):
        if start_line - 1 > prev_end:
            gap = "\n".join(lines[prev_end:start_line - 1]).strip()
            if gap:
                chunks.append((label, gap))

    for node in top_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _node_span(node)
            _flush_gap(start, "module-level")
            chunks.append((f"function {node.name}", "\n".join(lines[start - 1:end])))
            prev_end = end

        elif isinstance(node, ast.ClassDef):
            start, end = _node_span(node)
            _flush_gap(start, "module-level")
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if methods:
                m_prev = start - 1
                for m in methods:
                    m_start, m_end = _node_span(m)
                    if m_start - 1 > m_prev:
                        preamble = "\n".join(lines[m_prev:m_start - 1]).strip()
                        if preamble:
                            chunks.append((f"class {node.name} (body)", preamble))
                    chunks.append((f"class {node.name}.{m.name}", "\n".join(lines[m_start - 1:m_end])))
                    m_prev = m_end
                if end > m_prev:
                    tail = "\n".join(lines[m_prev:end]).strip()
                    if tail:
                        chunks.append((f"class {node.name} (body)", tail))
            else:
                chunks.append((f"class {node.name}", "\n".join(lines[start - 1:end])))
            prev_end = end
        # other top-level statements (imports, constants, `if __name__==...`)
        # are swept up into the next "module-level" gap flush.

    if prev_end < len(lines):
        tail = "\n".join(lines[prev_end:]).strip()
        if tail:
            chunks.append(("module-level", tail))

    return [(l, c) for l, c in chunks if c.strip()]


_JS_BOUNDARY = re.compile(
    r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s*\*?\s*\w*'
    r'|^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(?.*?\)?\s*=>'
    r'|^\s*(export\s+)?(abstract\s+)?class\s+\w+'
)


def _chunk_js_ts(text: str) -> list[tuple[str, str]]:
    """Heuristic boundary detection for JS/TS — splits on function/const-arrow/
    class declarations. Not a full parser, but avoids mid-function cuts."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _JS_BOUNDARY.match(line)]
    if not starts:
        return [("block", text)]

    chunks: list[tuple[str, str]] = []
    if starts[0] > 0:
        header = "\n".join(lines[:starts[0]]).strip()
        if header:
            chunks.append(("module-level", header))

    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        label = lines[s].strip()[:60] or "block"
        chunks.append((label, "\n".join(lines[s:e])))
    return chunks


def _chunk_fixed(text: str, size: int = CHUNK_SIZE) -> list[tuple[str, str]]:
    return [("part", text[i:i + size]) for i in range(0, len(text), size)]


def _chunk_file(path: str, text: str) -> list[str]:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "py":
        raw_chunks = _chunk_python(text)
    elif ext in {"js", "ts"}:
        raw_chunks = _chunk_js_ts(text)
    else:
        raw_chunks = _chunk_fixed(text)

    final: list[str] = []
    for label, code in raw_chunks:
        code = code.strip()
        if not code:
            continue
        if len(code) > MAX_CHUNK_CHARS:
            for _, sub in _chunk_fixed(code, MAX_CHUNK_CHARS):
                final.append(f"[FILE: {path} | {label}]\n{sub}")
        else:
            final.append(f"[FILE: {path} | {label}]\n{code}")
    return final


def build_faiss_index() -> faiss.IndexFlatL2:
    global _rag_chunks
    files = _fetch_repo_files()
    for path, raw in files:
        _rag_chunks.extend(_chunk_file(path, raw))
    embeddings = EMBED_MODEL.encode(_rag_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1] if len(embeddings) else 384)
    if len(embeddings):
        index.add(embeddings)
    return index


FAISS_INDEX = build_faiss_index()


def rag_search(query: str, k: int = 5) -> str:
    if not _rag_chunks:
        return ""
    q_vec = EMBED_MODEL.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    _, I = FAISS_INDEX.search(q_vec, k)
    results = [_rag_chunks[i] for i in I[0] if i < len(_rag_chunks)]
    return "\n---\n".join(results)


def wiki_search(query: str, sentences: int = 5) -> str:
    try:
        return wiki_lib.summary(query, sentences=sentences, auto_suggest=True)
    except wiki_lib.exceptions.DisambiguationError as e:
        try:
            return wiki_lib.summary(e.options[0], sentences=sentences)
        except Exception:
            return ""
    except Exception:
        return ""


class IssueState(TypedDict):
    issue_number : int
    title        : str
    body         : str
    raw_labels   : list
    issue_type   : Literal["bug", "documentation", "feature", "security", "unknown"]
    wiki_needed  : bool
    wiki_query   : str
    rag_context  : str
    wiki_context : str
    action_needed: str
    posted       : bool


def classifier_node(state: IssueState) -> IssueState:
    label_names = [(lbl.name if hasattr(lbl, "name") else str(lbl)).lower() for lbl in state["raw_labels"]]
    hint = ", ".join(label_names) if label_names else "none"

    prompt = f"""
        TYPE: <one of: bug | documentation | feature | security | unknown>

        Title : {state['title']}
        Body  : {state['body'][:800]}
        Labels: {hint}
    """
    resp = llm.invoke(prompt).content.strip()

    issue_type = "unknown"
    for line in resp.splitlines():
        if line.upper().startswith("TYPE:"):
            raw = line.split(":", 1)[1].strip().lower()
            if raw in {"bug", "documentation", "feature", "security"}:
                issue_type = raw

    return {**state, "issue_type": issue_type}


def rag_node(state: IssueState) -> IssueState:
    query   = f"{state['title']} {state['body'][:300]}"
    context = rag_search(query)
    return {**state, "rag_context": context}


# --------------------------------------------------------------------------
# 3. Wiki planning node — decides whether external Wikipedia background is
#    actually useful for this issue before spending a call on it, and if so,
#    generates the search phrase itself (replaces the old always-on lookup).
# --------------------------------------------------------------------------

def wiki_planner_node(state: IssueState) -> IssueState:
    prompt = f"""
        You are triaging a GitHub issue before writing a technical response.
        Decide whether looking up general background knowledge on Wikipedia
        would meaningfully help (e.g. the issue references a concept,
        algorithm, protocol, standard, or domain term worth grounding).
        Do NOT recommend a wiki lookup for issues that are purely about this
        codebase's internal logic, a specific stack trace, or a typo — RAG
        context already covers that.

        Respond in EXACTLY this format, nothing else:
        NEED_WIKI: <yes|no>
        KEYWORDS: <3-5 word search phrase, or NONE if NEED_WIKI is no>

        Issue type: {state['issue_type']}
        Title: {state['title']}
        Body: {state['body'][:800]}
    """
    resp = llm.invoke(prompt).content.strip()

    need = False
    keywords = ""
    for line in resp.splitlines():
        upper = line.upper()
        if upper.startswith("NEED_WIKI:"):
            need = line.split(":", 1)[1].strip().lower().startswith("y")
        elif upper.startswith("KEYWORDS:"):
            keywords = line.split(":", 1)[1].strip()

    if not need or keywords.upper() == "NONE":
        return {**state, "wiki_needed": False, "wiki_query": ""}
    return {**state, "wiki_needed": True, "wiki_query": keywords}


def _route_after_wiki_plan(state: IssueState) -> str:
    return "wiki" if state["wiki_needed"] else "specialist"


def wiki_node(state: IssueState) -> IssueState:
    context = wiki_search(state["wiki_query"])
    return {**state, "wiki_context": context}


_SPECIALIST_PROMPTS = {
    "bug": """
        1. Root Cause
        2. Debugging Steps
        3. Possible Fix
        4. Improvements
    """,
    "documentation": """
        1. Summary
        2. Missing or incorrect content
        3. Suggested revisions
        4. References
    """,
    "feature": """
        1. Summary
        2. Feasibility
        3. Implementation
        4. Challenges
        5. Criteria
    """,
    "security": """
        1. Issue Summary
        2. Impact
        3. Mitigation
        4. Long-term plan
    """,
    "unknown": """
        Provide a summary and recommendations.
    """,
}


def specialist_node(state: IssueState) -> IssueState:
    system = _SPECIALIST_PROMPTS[state["issue_type"]]
    user_msg = f"""
        {state['title']}
        {state['body'][:1200]}

        {state['rag_context'][:1500]}

        {state['wiki_context'][:800]}
    """
    resp = llm.invoke(f"{system}\n\n{user_msg}").content
    return {**state, "action_needed": resp}


def synthesizer_node(state: IssueState) -> IssueState:
    prompt = f"""
        Format the text below into clean markdown.
        {state['action_needed']}
    """
    refined = llm.invoke(prompt).content.strip()
    return {**state, "action_needed": refined}


def commenter_node(state: IssueState) -> IssueState:
    try:
        issue = repo.get_issue(state["issue_number"])
        body  = state["action_needed"]
        issue.create_comment(body)
        return {**state, "posted": True}
    except Exception:
        return {**state, "posted": False}


builder = StateGraph(IssueState)
builder.add_node("classifier",   classifier_node)
builder.add_node("rag",          rag_node)
builder.add_node("wiki_planner", wiki_planner_node)
builder.add_node("wiki",         wiki_node)
builder.add_node("specialist",   specialist_node)
builder.add_node("synthesizer",  synthesizer_node)
builder.add_node("commenter",    commenter_node)

builder.add_edge(START,          "classifier")
builder.add_edge("classifier",   "rag")
builder.add_edge("rag",          "wiki_planner")
builder.add_conditional_edges(
    "wiki_planner",
    _route_after_wiki_plan,
    {"wiki": "wiki", "specialist": "specialist"},
)
builder.add_edge("wiki",         "specialist")
builder.add_edge("specialist",   "synthesizer")
builder.add_edge("synthesizer",  "commenter")
builder.add_edge("commenter",    END)

app = builder.compile()


def run():
    issues = list(repo.get_issues(state="open"))
    for issue in issues:
        state: IssueState = {
            "issue_number" : issue.number,
            "title"        : issue.title,
            "body"         : issue.body or "",
            "raw_labels"   : list(issue.labels),
            "issue_type"   : "unknown",
            "wiki_needed"  : False,
            "wiki_query"   : "",
            "rag_context"  : "",
            "wiki_context" : "",
            "action_needed": "",
            "posted"       : False,
        }
        result = app.invoke(state)
        print(result["action_needed"], "\nPosted:", result["posted"], "\nWiki used:", result["wiki_needed"])


if __name__ == "__main__":
    run()
