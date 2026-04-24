"""
utils.py
--------
Utility helpers shared across the project.
Handles: logging setup, confidence scoring heuristic, and text cleaning.
"""

import logging
import re


def setup_logger(name: str = "rag_assistant") -> logging.Logger:
    """Set up a simple console logger for the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def estimate_confidence(answer: str, retrieved_docs: list) -> float:
    """
    Estimate a confidence score (0.0 → 1.0) based on simple heuristics:
      - Were any documents retrieved?
      - Does the answer contain uncertainty phrases?
      - Is the answer suspiciously short?

    In a production system you would use the LLM's own logprobs or a
    separate calibration call; this keeps things self-contained.
    """
    if not retrieved_docs:
        return 0.0  # No context at all → no confidence

    # Penalise answers that admit they don't know
    uncertainty_phrases = [
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "cannot find",
        "no information",
        "not mentioned",
        "not available",
        "unclear",
    ]
    answer_lower = answer.lower()
    for phrase in uncertainty_phrases:
        if phrase in answer_lower:
            return 0.2  # Very low confidence

    # Short answers (< 20 chars) are usually non-answers
    if len(answer.strip()) < 20:
        return 0.3

    # Base confidence when docs exist and answer looks substantive
    base = 0.75

    # Bonus: more retrieved docs → slightly higher confidence
    doc_bonus = min(len(retrieved_docs) * 0.05, 0.20)

    return round(min(base + doc_bonus, 1.0), 2)


def clean_text(text: str) -> str:
    """Strip extra whitespace and normalise line breaks."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()
