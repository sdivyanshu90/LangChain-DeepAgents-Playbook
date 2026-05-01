from __future__ import annotations

from pathlib import Path
import operator
from typing import Annotated, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import Settings


IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv"}


def inventory_repo(repo_path: str, limit: int = 50) -> list[str]:
    root = Path(repo_path)
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.is_file():
            entries.append(path.relative_to(root).as_posix())
        if len(entries) >= limit:
            break
    return entries


def summarize_inventory_modules(inventory: list[str]) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for item in inventory:
        module = item.split("/", 1)[0]
        grouped.setdefault(module, []).append(item)

    summaries = []
    for module, files in sorted(grouped.items()):
        preview = ", ".join(files[:5])
        summaries.append(f"Module: {module}\nFiles: {preview}")
    return summaries


def parse_files_from_summary(summary: str) -> list[str]:
    for line in summary.splitlines():
        if line.startswith("Files: "):
            return [item.strip() for item in line.removeprefix("Files: ").split(",") if item.strip()]
    return []


def build_index(summaries: list[str], settings: Settings) -> FAISS:
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    return FAISS.from_texts(summaries, embeddings)


def route_after_scan(state: ExplorerState) -> str:
    if state.get("current_question"):
        return "select"
    return "summarize_modules"


class ExplorerState(TypedDict, total=False):
    repo_path: str
    current_question: str
    inventory: Annotated[list[str], operator.add]
    module_summaries: Annotated[list[str], operator.add]
    selected_files: Annotated[list[str], operator.add]
    file_contents: Annotated[list[str], operator.add]
    final_answer: str


def build_app(settings: Settings):
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You explain repository structure based only on the provided inventory and file excerpts."),
            ("human", "Question: {question}\n\nInventory:\n{inventory}\n\nSelected file excerpts:\n{file_contents}"),
        ]
    )
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You summarize a repository architecture from module summaries."),
            ("human", "Repository inventory:\n{inventory}\n\nModule summaries:\n{module_summaries}"),
        ]
    )
    model = ChatOpenAI(model=settings.model, temperature=settings.temperature)
    explainer = answer_prompt | model | StrOutputParser()
    summarizer = summary_prompt | model | StrOutputParser()

    def inventory_node(state: ExplorerState) -> ExplorerState:
        inventory = inventory_repo(state["repo_path"])
        return {"inventory": inventory, "module_summaries": summarize_inventory_modules(inventory)}

    def select_node(state: ExplorerState) -> ExplorerState:
        index = build_index(state.get("module_summaries", []), settings)
        relevant = index.similarity_search(state["current_question"], k=min(3, len(state.get("module_summaries", []))))
        selected_files: list[str] = []
        for document in relevant:
            for file_path in parse_files_from_summary(document.page_content):
                if file_path not in selected_files:
                    selected_files.append(file_path)
        if not selected_files:
            selected_files = state.get("inventory", [])[:5]
        return {"selected_files": selected_files[:5]}

    def read_node(state: ExplorerState) -> ExplorerState:
        root = Path(state["repo_path"])
        contents = []
        for relative_path in state.get("selected_files", []):
            path = root / relative_path
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            contents.append(f"[{relative_path}]\n{text[:1200]}")
        return {"file_contents": contents}

    def answer_node(state: ExplorerState) -> ExplorerState:
        answer = explainer.invoke(
            {
                "question": state["current_question"],
                "inventory": "\n".join(state.get("inventory", [])),
                "file_contents": "\n\n".join(state.get("file_contents", [])),
            }
        )
        return {"final_answer": answer}

    def summarize_modules_node(state: ExplorerState) -> ExplorerState:
        answer = summarizer.invoke(
            {
                "inventory": "\n".join(state.get("inventory", [])),
                "module_summaries": "\n\n".join(state.get("module_summaries", [])),
            }
        )
        return {"final_answer": answer}

    graph = StateGraph(ExplorerState)
    graph.add_node("inventory", inventory_node)
    graph.add_node("select", select_node)
    graph.add_node("read", read_node)
    graph.add_node("answer", answer_node)
    graph.add_node("summarize_modules", summarize_modules_node)
    graph.add_edge(START, "inventory")
    graph.add_conditional_edges("inventory", route_after_scan, {"select": "select", "summarize_modules": "summarize_modules"})
    graph.add_edge("select", "read")
    graph.add_edge("read", "answer")
    graph.add_edge("summarize_modules", END)
    graph.add_edge("answer", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
