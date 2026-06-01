import os, textwrap, re
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
CHUNK_SIZE  = 400
_rag_chunks = []

def _fetch_repo_files(max_files: int = 60) -> list[str]:
    texts = []
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
                if ext in {"py","md","txt","js","ts","json","yaml","yml","html","css","rst","ipynb","sh"}:
                    try:
                        raw = item.decoded_content.decode("utf-8", errors="ignore")
                        texts.append(f"[FILE: {item.path}]\n{raw}")
                        visited += 1
                    except Exception:
                        pass
    except Exception:
        pass
    return texts

def _chunk(text: str, size: int = CHUNK_SIZE) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]

def build_faiss_index() -> faiss.IndexFlatL2:
    global _rag_chunks
    files = _fetch_repo_files()
    for f in files:
        _rag_chunks.extend(_chunk(f))
    embeddings = EMBED_MODEL.encode(_rag_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

FAISS_INDEX = build_faiss_index()

def rag_search(query: str, k: int = 5) -> str:
    if not _rag_chunks:
        return ""
    q_vec = EMBED_MODEL.encode([query], show_progress_bar=False)
    q_vec = np.array(q_vec, dtype="float32")
    _, I  = FAISS_INDEX.search(q_vec, k)
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
        WIKI: <3–5 word phrase>

        Title : {state['title']}
        Body  : {state['body'][:800]}
        Labels: {hint}
    """
    resp = llm.invoke(prompt).content.strip()

    issue_type = "unknown"
    wiki_query = state["title"]

    for line in resp.splitlines():
        if line.upper().startswith("TYPE:"):
            raw = line.split(":", 1)[1].strip().lower()
            if raw in {"bug","documentation","feature","security"}:
                issue_type = raw
        elif line.upper().startswith("WIKI:"):
            wiki_query = line.split(":", 1)[1].strip()

    return {**state, "issue_type": issue_type, "wiki_query": wiki_query}

def rag_node(state: IssueState) -> IssueState:
    query   = f"{state['title']} {state['body'][:300]}"
    context = rag_search(query)
    return {**state, "rag_context": context}

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
builder.add_node("classifier", classifier_node)
builder.add_node("rag",        rag_node)
builder.add_node("wiki",       wiki_node)
builder.add_node("specialist", specialist_node)
builder.add_node("synthesizer",synthesizer_node)
builder.add_node("commenter",  commenter_node)

builder.add_edge(START,        "classifier")
builder.add_edge("classifier", "rag")
builder.add_edge("rag",        "wiki")
builder.add_edge("wiki",       "specialist")
builder.add_edge("specialist", "synthesizer")
builder.add_edge("synthesizer","commenter")
builder.add_edge("commenter",  END)

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
            "wiki_query"   : "",
            "rag_context"  : "",
            "wiki_context" : "",
            "action_needed": "",
            "posted"       : False,
        }
        result = app.invoke(state)
        print(result["action_needed"], "\nPosted:", result["posted"])

if __name__ == "__main__":
    run()