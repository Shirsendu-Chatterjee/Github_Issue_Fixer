import os, re, ast, sys, json
from typing import TypedDict, Literal
from langchain_groq import ChatGroq
import github as gh_lib
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
import wikipedia as wiki_lib
from langgraph.graph import StateGraph, START, END

os.environ["GROQ_API_KEY"] = ""
GITHUB_TOKEN = ""
REPO_NAME    = ""

llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.2)

auth = gh_lib.Auth.Token(GITHUB_TOKEN)
g    = gh_lib.Github(auth=auth)
repo = g.get_repo(REPO_NAME)

wiki_lib.set_lang("en")

# Code-search-tuned encoder (trained on CodeSearchNet: docstring<->code pairs)
# instead of general-purpose MiniLM — much better at matching natural-language
# issue text to code semantics.
EMBED_MODEL = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")

CHUNK_SIZE          = 400   # fallback char-chunk size for non-code files
MAX_CHUNK_CHARS     = 1200  # any single function/class chunk larger than this gets sub-split
MAX_RETRIEVAL_HOPS  = 3     # cap on agentic retrieve->replan loops

_rag_chunks: list[str] = []   # embedded/stored text, aligned index-for-index with _rag_labels
_rag_labels: list[str] = []   # e.g. "app.py | class Foo.method_a" — for showing what was retrieved


# --------------------------------------------------------------------------
# Repo fetching
# --------------------------------------------------------------------------

def _fetch_repo_files(max_files: int = 60) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    try:
        queue = list(repo.get_contents(""))
        visited = 0
        while queue and visited < max_files:
            item = queue.pop(0)
            if item.type == "dir":
                queue.extend(repo.get_contents(item.path))
                continue
            ext = item.path.rsplit(".", 1)[-1].lower()
            if item.type == "file" and item.size < 80_000 and ext in {
                "py", "md", "txt", "js", "ts", "json", "yaml", "yml", "html", "css", "rst", "ipynb", "sh"
            }:
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
# Function/class-aware chunking (AST for .py, regex boundaries for .js/.ts,
# fixed-size fallback for everything else).
# --------------------------------------------------------------------------

def _node_span(node: ast.AST) -> tuple[int, int]:
    start = node.lineno
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        start = min(start, decorators[0].lineno)
    return start, getattr(node, "end_lineno", node.lineno)


def _chunk_python(text: str) -> list[tuple[str, str]]:
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

    def flush_gap(upto_line: int, label: str):
        if upto_line - 1 > prev_end:
            gap = "\n".join(lines[prev_end:upto_line - 1]).strip()
            if gap:
                chunks.append((label, gap))

    for node in top_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _node_span(node)
            flush_gap(start, "module-level")
            chunks.append((f"function {node.name}", "\n".join(lines[start - 1:end])))
            prev_end = end

        elif isinstance(node, ast.ClassDef):
            start, end = _node_span(node)
            flush_gap(start, "module-level")
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
        chunks.append((lines[s].strip()[:60] or "block", "\n".join(lines[s:e])))
    return chunks


def _chunk_fixed(text: str, size: int = CHUNK_SIZE) -> list[tuple[str, str]]:
    return [("part", text[i:i + size]) for i in range(0, len(text), size)]


def _split_by_extension(path: str, text: str) -> list[tuple[str, str]]:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "py":
        return _chunk_python(text)
    if ext in {"js", "ts"}:
        return _chunk_js_ts(text)
    return _chunk_fixed(text)


# --------------------------------------------------------------------------
# No LLM call here on purpose — summarizing every chunk before embedding
# burns one LLM call per function/class in the repo (token cost scales with
# repo size). Store raw code, tagged with its file + function/class label,
# and let the code-search-tuned encoder handle semantic matching directly.
# --------------------------------------------------------------------------

def _chunk_file(path: str, text: str) -> list[tuple[str, str]]:
    """Returns (label, embedded_text) pairs — label is used later to show
    which function/class/block was actually retrieved."""
    final: list[tuple[str, str]] = []
    for label, code in _split_by_extension(path, text):
        code = code.strip()
        if not code:
            continue
        pieces = [code] if len(code) <= MAX_CHUNK_CHARS else [c for _, c in _chunk_fixed(code, MAX_CHUNK_CHARS)]
        for piece in pieces:
            tag = f"{path} | {label}"
            final.append((tag, f"[FILE: {tag}]\n{piece}"))
    return final


def build_faiss_index() -> faiss.IndexFlatL2:
    """Fetches the repo, chunks every file, and embeds everything from
    scratch. Only called on a cache miss — see load_or_build_index."""
    global _rag_chunks, _rag_labels
    for path, raw in _fetch_repo_files():
        for label, text in _chunk_file(path, raw):
            _rag_labels.append(label)
            _rag_chunks.append(text)

    dim = EMBED_MODEL.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    if _rag_chunks:
        embeddings = np.array(EMBED_MODEL.encode(_rag_chunks, show_progress_bar=False), dtype="float32")
        index.add(embeddings)
    return index


# --------------------------------------------------------------------------
# Persistence: one folder per repo under ./database, e.g.
#   database/Shirsendu-Chatterjee__ei/index.faiss   (the vectors)
#   database/Shirsendu-Chatterjee__ei/chunks.json    (text + labels, index-aligned)
# On startup we check for this folder first — only fetch/chunk/embed the
# repo (the expensive path) if no cache exists yet.
# --------------------------------------------------------------------------

DB_ROOT = "database"


def _repo_db_dir(repo_name: str) -> str:
    return os.path.join(DB_ROOT, repo_name.replace("/", "__"))


def load_or_build_index(repo_name: str) -> faiss.IndexFlatL2:
    global _rag_chunks, _rag_labels
    db_dir     = _repo_db_dir(repo_name)
    index_path = os.path.join(db_dir, "index.faiss")
    meta_path  = os.path.join(db_dir, "chunks.json")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        _rag_chunks = meta["chunks"]
        _rag_labels = meta["labels"]
        print(f"[rag] loaded {len(_rag_chunks)} cached chunks from {db_dir}")
        return faiss.read_index(index_path)

    print(f"[rag] no cache for {repo_name!r} — indexing repo from scratch")
    index = build_faiss_index()
    os.makedirs(db_dir, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": _rag_chunks, "labels": _rag_labels}, f)
    print(f"[rag] cached {len(_rag_chunks)} chunks to {db_dir}")
    return index


FAISS_INDEX = load_or_build_index(REPO_NAME)


def rag_search(query: str, k: int = 5) -> str:
    if not _rag_chunks:
        return ""
    q_vec = np.array(EMBED_MODEL.encode([query], show_progress_bar=False), dtype="float32")
    _, indices = FAISS_INDEX.search(q_vec, k)
    valid = [i for i in indices[0] if 0 <= i < len(_rag_chunks)]

    if valid:
        print(f"[retrieve] query={query!r}")
        for i in valid:
            print(f"    -> {_rag_labels[i]}")

    return "\n---\n".join(_rag_chunks[i] for i in valid)


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
    issue_number    : int
    title           : str
    body            : str
    raw_labels      : list
    issue_type      : Literal["bug", "documentation", "feature", "security", "unknown"]
    rag_context     : str
    queries_tried   : list
    retrieval_hops  : int
    next_query      : str
    wiki_needed     : bool
    wiki_query      : str
    wiki_context    : str
    action_needed   : str
    posted          : bool


def classifier_node(state: IssueState) -> IssueState:
    labels = [(lbl.name if hasattr(lbl, "name") else str(lbl)).lower() for lbl in state["raw_labels"]]
    prompt = f"""
        TYPE: <one of: bug | documentation | feature | security | unknown>

        Title : {state['title']}
        Body  : {state['body'][:800]}
        Labels: {", ".join(labels) or "none"}
    """
    resp = llm.invoke(prompt).content.strip()
    issue_type = "unknown"
    for line in resp.splitlines():
        if line.upper().startswith("TYPE:"):
            raw = line.split(":", 1)[1].strip().lower()
            if raw in {"bug", "documentation", "feature", "security"}:
                issue_type = raw
    return {**state, "issue_type": issue_type}


# --------------------------------------------------------------------------
# Agentic retrieval loop: retrieve -> ask the LLM "is this enough, or what
# should we search next" -> retrieve again if needed, up to MAX_RETRIEVAL_HOPS.
# --------------------------------------------------------------------------

def retrieve_node(state: IssueState) -> IssueState:
    query = state["next_query"] or f"{state['title']} {state['body'][:300]}"
    result = rag_search(query)

    context = state["rag_context"]
    if result:
        context = f"{context}\n---\n{result}" if context else result

    return {
        **state,
        "rag_context": context,
        "queries_tried": state["queries_tried"] + [query],
        "retrieval_hops": state["retrieval_hops"] + 1,
    }


def retrieval_planner_node(state: IssueState) -> IssueState:
    if state["retrieval_hops"] >= MAX_RETRIEVAL_HOPS:
        return {**state, "next_query": ""}

    prompt = f"""
        You're deciding whether enough repo context has been retrieved to
        write a technical response to this GitHub issue, or whether another,
        DIFFERENTLY WORDED search would surface something important that's
        missing.

        Title: {state['title']}
        Body : {state['body'][:500]}

        Queries already tried: {state['queries_tried']}
        Retrieved so far:
        {state['rag_context'][:2000]}

        Respond in EXACTLY this format:
        ENOUGH: <yes|no>
        NEXT_QUERY: <a new, more specific search query, or NONE if ENOUGH is yes>
    """
    resp = llm.invoke(prompt).content.strip()

    enough, next_query = True, ""
    for line in resp.splitlines():
        upper = line.upper()
        if upper.startswith("ENOUGH:"):
            enough = line.split(":", 1)[1].strip().lower().startswith("y")
        elif upper.startswith("NEXT_QUERY:"):
            next_query = line.split(":", 1)[1].strip()

    if enough or next_query.upper() == "NONE":
        next_query = ""
    return {**state, "next_query": next_query}


def _route_after_retrieval_plan(state: IssueState) -> str:
    return "retrieve" if state["next_query"] else "wiki_planner"


def wiki_planner_node(state: IssueState) -> IssueState:
    prompt = f"""
        Decide whether looking up general background knowledge on Wikipedia
        would meaningfully help answer this GitHub issue (e.g. it references
        a concept, protocol, algorithm, or standard worth grounding). Do NOT
        recommend it for issues that are purely about this codebase's
        internal logic or a specific stack trace — repo context already
        covers that.

        Respond in EXACTLY this format:
        NEED_WIKI: <yes|no>
        KEYWORDS: <3-5 word search phrase, or NONE if NEED_WIKI is no>

        Issue type: {state['issue_type']}
        Title: {state['title']}
        Body: {state['body'][:800]}
    """
    resp = llm.invoke(prompt).content.strip()

    need, keywords = False, ""
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
    return {**state, "wiki_context": wiki_search(state["wiki_query"])}


_SPECIALIST_PROMPTS = {
    "bug": "1. Root Cause\n2. Debugging Steps\n3. Possible Fix\n4. Improvements",
    "documentation": "1. Summary\n2. Missing or incorrect content\n3. Suggested revisions\n4. References",
    "feature": "1. Summary\n2. Feasibility\n3. Implementation\n4. Challenges\n5. Criteria",
    "security": "1. Issue Summary\n2. Impact\n3. Mitigation\n4. Long-term plan",
    "unknown": "Provide a summary and recommendations.",
}


def specialist_node(state: IssueState) -> IssueState:
    system = _SPECIALIST_PROMPTS[state["issue_type"]]
    user_msg = f"""
        {state['title']}
        {state['body'][:1200]}

        {state['rag_context'][:2500]}

        {state['wiki_context'][:800]}
    """
    resp = llm.invoke(f"{system}\n\n{user_msg}").content
    return {**state, "action_needed": resp}


def synthesizer_node(state: IssueState) -> IssueState:
    refined = llm.invoke(f"Format the text below into clean markdown.\n{state['action_needed']}").content.strip()
    return {**state, "action_needed": refined}


def commenter_node(state: IssueState) -> IssueState:
    try:
        repo.get_issue(state["issue_number"]).create_comment(state["action_needed"])
        return {**state, "posted": True}
    except Exception:
        return {**state, "posted": False}


builder = StateGraph(IssueState)
builder.add_node("classifier",        classifier_node)
builder.add_node("retrieve",          retrieve_node)
builder.add_node("retrieval_planner", retrieval_planner_node)
builder.add_node("wiki_planner",      wiki_planner_node)
builder.add_node("wiki",              wiki_node)
builder.add_node("specialist",        specialist_node)
builder.add_node("synthesizer",       synthesizer_node)
builder.add_node("commenter",         commenter_node)

builder.add_edge(START,               "classifier")
builder.add_edge("classifier",        "retrieve")
builder.add_edge("retrieve",          "retrieval_planner")
builder.add_conditional_edges(
    "retrieval_planner",
    _route_after_retrieval_plan,
    {"retrieve": "retrieve", "wiki_planner": "wiki_planner"},
)
builder.add_conditional_edges(
    "wiki_planner",
    _route_after_wiki_plan,
    {"wiki": "wiki", "specialist": "specialist"},
)
builder.add_edge("wiki",              "specialist")
builder.add_edge("specialist",        "synthesizer")
builder.add_edge("synthesizer",       "commenter")
builder.add_edge("commenter",         END)

app = builder.compile()


def run(issue_number: int):
    """Processes exactly one issue — deliberately not a loop over all open
    issues, since each run can involve several LLM calls (classification,
    multiple retrieval hops, wiki planning, specialist reasoning, synthesis)
    and looping over every open issue would multiply token usage fast."""
    issue = repo.get_issue(issue_number)
    state: IssueState = {
        "issue_number"  : issue.number,
        "title"         : issue.title,
        "body"          : issue.body or "",
        "raw_labels"    : list(issue.labels),
        "issue_type"    : "unknown",
        "rag_context"   : "",
        "queries_tried" : [],
        "retrieval_hops": 0,
        "next_query"    : "",
        "wiki_needed"   : False,
        "wiki_query"    : "",
        "wiki_context"  : "",
        "action_needed" : "",
        "posted"        : False,
    }
    result = app.invoke(state)
    print(result["action_needed"])
    print("Posted:", result["posted"], "| retrieval hops:", result["retrieval_hops"], "| wiki used:", result["wiki_needed"])
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python issue_fixer.py <issue_number>")
    run(int(sys.argv[1]))
