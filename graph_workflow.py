"""
graph_workflow.py
-----------------
Defines the LangGraph-based workflow for the RAG Customer Support Assistant.

Graph topology
==============

  [START]
     │
     ▼
  ┌──────────────────┐
  │  processing_node │  ← retrieval + answer generation
  └──────────────────┘
     │
     ▼
  ┌──────────────────┐
  │  decision_node   │  ← checks confidence & escalation_flag
  └──────────────────┘
     │
     ├─── "escalate" ──► ┌──────────────────┐
     │                   │  escalation_node │  ← HITL handler
     │                   └──────────────────┘
     │                              │
     └─── "output" ────► ┌──────────────────┐ ◄──┘
                         │   output_node    │  ← returns final answer
                         └──────────────────┘
                                    │
                                  [END]

State object
============
The shared state dict carries:
  query           : str   — user's original question
  retrieved_docs  : list  — Document chunks from ChromaDB
  answer          : str   — LLM-generated (or human-provided) answer
  confidence      : float — heuristic confidence in the answer [0, 1]
  escalation_flag : bool  — True if human escalation is needed
  escalation_reason: str  — Human-readable reason for escalation

Dependencies:
  pip install langgraph
"""

from typing import TypedDict, List, Annotated
import operator

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from rag_pipeline import retrieve_docs, generate_answer
from hitl import escalate_to_human
from utils import setup_logger, estimate_confidence

logger = setup_logger()

# ---------------------------------------------------------------------------
# Confidence threshold — answers below this score will be escalated
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class SupportState(TypedDict):
    """Shared state object passed between every node in the graph."""
    query            : str
    retrieved_docs   : List[Document]
    answer           : str
    confidence       : float
    escalation_flag  : bool
    escalation_reason: str


# ---------------------------------------------------------------------------
# Node 1: processing_node
# Retrieves relevant chunks and generates a candidate answer.
# ---------------------------------------------------------------------------

def processing_node(state: SupportState, vector_store) -> SupportState:
    """
    1. Run similarity search to find relevant chunks.
    2. Send chunks + query to the LLM and get a candidate answer.
    3. Estimate confidence.
    4. Set escalation_flag if no docs were found or similarity was too low.
    """
    query = state["query"]
    logger.info("[processing_node] Handling query: '%s'", query)

    # --- Retrieval ---
    docs, best_score = retrieve_docs(vector_store, query)

    # --- Escalation conditions that can be detected pre-LLM ---
    if not docs:
        return {
            **state,
            "retrieved_docs"   : [],
            "answer"           : "",
            "confidence"       : 0.0,
            "escalation_flag"  : True,
            "escalation_reason": "No relevant documents found in the knowledge base.",
        }

    # Always generate answer if docs exist
    answer = generate_answer(query, docs)

    # Use similarity score directly as confidence base
    confidence = max(best_score, 0.0)

    logger.info("[processing_node] Confidence: %.2f", confidence)

    return {
        **state,
        "retrieved_docs"   : docs,
        "answer"           : answer,
        "confidence"       : confidence,
        "escalation_flag"  : False,
        "escalation_reason": "",
    }


# ---------------------------------------------------------------------------
# Node 2: decision_node
# Applies routing logic: escalate or pass through to output.
# ---------------------------------------------------------------------------

def decision_node(state: SupportState) -> SupportState:
    """
    Examine the current state and decide whether to escalate.
    This node only *updates* the escalation fields; the actual routing
    is performed by the conditional edge (route_after_decision).
    """
    logger.info("[decision_node] confidence=%.2f escalation_flag=%s",
                state["confidence"], state["escalation_flag"])

    # Already flagged for escalation upstream → keep the flag
    if state["escalation_flag"]:
        return state

    # Escalate if confidence is below threshold
    confidence = state["confidence"]
    answer = state["answer"]

    is_low_confidence = confidence < CONFIDENCE_THRESHOLD
    is_unknown = "i don't know" in answer.lower()

    if is_low_confidence or is_unknown:
        return {
            **state,
            "escalation_flag": True,
            "escalation_reason": (
                "Low confidence or insufficient knowledge to answer the query."
            ),
        }

    # All checks passed → no escalation needed
    return state


# ---------------------------------------------------------------------------
# Conditional edge: determines which node comes after decision_node
# ---------------------------------------------------------------------------

def route_after_decision(state: SupportState) -> str:
    """
    Return the name of the next node based on escalation_flag.
    LangGraph uses this return value to follow the correct edge.
    """
    if state["escalation_flag"]:
        logger.info("[router] Routing → escalation_node")
        return "escalate"
    logger.info("[router] Routing → output_node")
    return "output"


# ---------------------------------------------------------------------------
# Node 3a: escalation_node
# Calls the HITL module to get a human response.
# ---------------------------------------------------------------------------

def escalation_node(state: SupportState) -> SupportState:
    """
    Trigger the Human-in-the-Loop handler and store the human's answer
    back into the state so output_node can display it uniformly.
    """
    logger.info("[escalation_node] Escalating to human agent.")

    human_answer = escalate_to_human(
        query=state["query"],
        reason=state["escalation_reason"],
        use_input=False,  # Set to True to collect a live human response
    )

    return {
        **state,
        "answer": human_answer,
    }


# ---------------------------------------------------------------------------
# Node 3b: output_node
# Final node — just logs and passes the answer through.
# ---------------------------------------------------------------------------

def output_node(state: SupportState) -> SupportState:
    """
    Format and return the final answer.
    In a production system this would also write to a database, send an
    e-mail, call a webhook, etc.
    """
    logger.info("[output_node] Returning answer to user.")
    print("\n" + "=" * 60)
    print("✅  ANSWER")
    print("=" * 60)
    print(state["answer"])
    print("=" * 60 + "\n")
    return state


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(vector_store):
    """
    Assemble and compile the LangGraph StateGraph.

    We use a closure so that processing_node has access to `vector_store`
    without it being part of the shared state.
    """

    # Wrap processing_node so it receives vector_store via closure
    def _processing_node(state: SupportState) -> SupportState:
        return processing_node(state, vector_store)

    graph = StateGraph(SupportState)

    # --- Register nodes ---
    graph.add_node("processing_node" , _processing_node)
    graph.add_node("decision_node"   , decision_node)
    graph.add_node("escalation_node" , escalation_node)
    graph.add_node("output_node"     , output_node)

    # --- Define edges ---
    graph.set_entry_point("processing_node")
    graph.add_edge("processing_node", "decision_node")

    # Conditional edge: decision_node → escalation_node OR output_node
    graph.add_conditional_edges(
        "decision_node",
        route_after_decision,
        {
            "escalate": "escalation_node",
            "output"  : "output_node",
        },
    )

    # Both paths converge at END
    graph.add_edge("escalation_node", "output_node")
    graph.add_edge("output_node"    , END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Public helper: run the workflow for a single query
# ---------------------------------------------------------------------------

def run_workflow(compiled_graph, query: str) -> SupportState:
    """
    Execute the compiled graph for one user query.

    Parameters
    ----------
    compiled_graph : CompiledGraph returned by build_graph()
    query          : The user's support question.

    Returns
    -------
    SupportState dict with all fields populated after the run.
    """
    initial_state: SupportState = {
        "query"            : query,
        "retrieved_docs"   : [],
        "answer"           : "",
        "confidence"       : 0.0,
        "escalation_flag"  : False,
        "escalation_reason": "",
    }

    logger.info("Starting workflow for query: '%s'", query)
    final_state = compiled_graph.invoke(initial_state)
    logger.info("Workflow complete. Escalated: %s", final_state["escalation_flag"])
    return final_state
