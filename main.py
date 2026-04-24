"""
main.py
-------
Entry point for the RAG Customer Support Assistant.

Run modes
---------
  python main.py --build           Build (or rebuild) the vector store from the PDF.
  python main.py --query "text"    Run a single query against the existing vector store.
  python main.py                   Interactive REPL mode (type queries until 'exit').

Environment variables required
-------------------------------
  OPENAI_API_KEY   — your OpenAI key

Optional overrides (see rag_pipeline.py for defaults)
------------------------------------------------------
  PDF_PATH, CHROMA_PERSIST_DIR, LLM_MODEL, EMBEDDING_MODEL, TOP_K, SIM_THRESHOLD
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# Make sure an API key is present before we import LangChain modules
# ---------------------------------------------------------------------------

from rag_pipeline import load_and_split_pdf, build_vector_store, load_vector_store
from graph_workflow import build_graph, run_workflow
from utils import setup_logger

logger = setup_logger()

# ---------------------------------------------------------------------------
# Default path to the PDF knowledge base
# Override with:  export PDF_PATH=/path/to/your/manual.pdf
# ---------------------------------------------------------------------------
DEFAULT_PDF_PATH = os.getenv("PDF_PATH", "knowledge_base.pdf")


# ---------------------------------------------------------------------------
# Helper: print a short summary of the final workflow state
# ---------------------------------------------------------------------------

def print_summary(state: dict) -> None:
    print("\n--- Run Summary ---")
    print(f"  Query      : {state['query']}")
    print(f"  Confidence : {state['confidence']:.2f}")
    print(f"  Escalated  : {state['escalation_flag']}")
    if state["escalation_reason"]:
        print(f"  Reason     : {state['escalation_reason']}")
    print("-------------------\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG Customer Support Assistant (LangGraph + HITL)"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="(Re)build the ChromaDB vector store from the PDF knowledge base.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query and exit.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Build vector store (--build flag or first run)
    # -----------------------------------------------------------------------
    if args.build:
        logger.info("Build mode: loading and indexing PDF…")
        chunks       = load_and_split_pdf(DEFAULT_PDF_PATH)
        vector_store = build_vector_store(chunks)
        logger.info("Vector store built. Re-run without --build to query.")
        return

    # -----------------------------------------------------------------------
    # Step 2: Load existing vector store
    # -----------------------------------------------------------------------
    try:
        vector_store = load_vector_store()
    except Exception as exc:
        logger.error(
            "Could not load vector store: %s\n"
            "Run with --build first to index your PDF.",
            exc,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Compile the LangGraph workflow
    # -----------------------------------------------------------------------
    compiled_graph = build_graph(vector_store)

    # -----------------------------------------------------------------------
    # Step 4: Single-query mode or interactive REPL
    # -----------------------------------------------------------------------
    if args.query:
        # --- Single query ---
        state = run_workflow(compiled_graph, args.query)
        print_summary(state)
    else:
        # --- Interactive REPL ---
        print("\n" + "=" * 60)
        print("  RAG Customer Support Assistant  (type 'exit' to quit)")
        print("=" * 60 + "\n")

        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            state = run_workflow(compiled_graph, query)
            print_summary(state)


if __name__ == "__main__":
    main()
