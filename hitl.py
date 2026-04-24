"""
hitl.py
-------
Human-in-the-Loop (HITL) module.
When the LangGraph workflow decides to escalate, this module takes over:
  1. Notifies the human agent.
  2. Optionally accepts a live human response (or returns a simulated one).
"""

from utils import setup_logger

logger = setup_logger()

# ---------------------------------------------------------------------------
# Simulated human-agent response bank
# (In production, replace with a ticketing system, Slack webhook, e-mail, etc.)
# ---------------------------------------------------------------------------
SIMULATED_RESPONSES = {
    "default": (
        "Thank you for reaching out. A human support agent has reviewed your query. "
        "Please allow 1–2 business days for a detailed response. "
        "In the meantime, you can visit our Help Centre at https://support.example.com."
    ),
    "refund": (
        "Our refund policy allows returns within 30 days of purchase. "
        "Please e-mail refunds@example.com with your order number and we will process it promptly."
    ),
    "technical": (
        "Our engineering team has been notified about this technical issue. "
        "A specialist will contact you within 4 business hours."
    ),
}


def escalate_to_human(query: str, reason: str, use_input: bool = False) -> str:
    """
    Escalate a query to a human agent.

    Parameters
    ----------
    query       : The original user question.
    reason      : Why escalation was triggered (low confidence, no docs, etc.).
    use_input   : If True, prompt the terminal for a real human reply.
                  If False, return a canned simulated response.

    Returns
    -------
    str : The human agent's response.
    """
    print("\n" + "=" * 60)
    print("🚨  ESCALATING TO HUMAN AGENT")
    print("=" * 60)
    print(f"  Reason  : {reason}")
    print(f"  Query   : {query}")
    print("=" * 60)

    logger.warning("Escalation triggered | reason='%s' | query='%s'", reason, query)

    if use_input:
        # --- Live mode: let a real person type the answer ---
        print("\n[Human Agent] Please type your response and press Enter:")
        human_response = input(">>> ").strip()
        if not human_response:
            human_response = SIMULATED_RESPONSES["default"]
    else:
        # --- Demo mode: pick a canned response based on keywords ---
        query_lower = query.lower()
        if any(w in query_lower for w in ["refund", "return", "money back"]):
            human_response = SIMULATED_RESPONSES["refund"]
        elif any(w in query_lower for w in ["error", "bug", "crash", "not working", "technical"]):
            human_response = SIMULATED_RESPONSES["technical"]
        else:
            human_response = SIMULATED_RESPONSES["default"]

    print(f"\n[Human Agent Response]\n{human_response}\n")
    return human_response
