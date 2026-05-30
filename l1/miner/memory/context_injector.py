"""
Formats prior ThoughtResults into a <prior_thinking> XML context block
and prepends it to the inference prompt.

The context block gives the model access to relevant prior thinking from other
miners, enabling distributed memory across the inference network.
"""

from __future__ import annotations

import logging

log = logging.getLogger("context_injector")

# ── Token estimation ──────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters (common rule of thumb)."""
    return max(1, len(text) // 4)


# ── XML context formatter ─────────────────────────────────────────────────────

def format_prior_context(results: list[dict], max_tokens: int = 512) -> str:
    """
    Format a list of ThoughtResult dicts into a <prior_thinking> XML block.

    Each result becomes an <entry> element with similarity score, model, and
    a truncated miner address (first 10 chars + "...") as attributes, and
    <question>, <thinking>, and <answer> sub-elements.

    Token budget enforcement:
      - Estimate tokens as len(text) // 4.
      - Thinking text is truncated (and marked with "...") to fit within
        the remaining token budget.
      - Answer text is preserved with higher priority than thinking text:
        if we must truncate, thinking is cut first.
      - Entries are skipped entirely if even the skeleton (without thinking)
        would exceed the remaining budget.

    Returns '' if results is empty or all entries are too large to include.
    """
    if not results:
        return ""

    # Fixed XML overhead per entry (tags, attributes template)
    ENTRY_SKELETON = (
        "  <entry similarity=\"{score:.2f}\" model=\"{model_id}\" miner=\"{miner}...\">\n"
        "    <question>{question}</question>\n"
        "    <thinking></thinking>\n"
        "    <answer>{answer}</answer>\n"
        "  </entry>\n"
    )
    OUTER_OPEN  = "<prior_thinking>\n"
    OUTER_CLOSE = "</prior_thinking>"

    # Budget available for content (subtract outer tags overhead)
    outer_tokens = _estimate_tokens(OUTER_OPEN + OUTER_CLOSE)
    budget = max_tokens - outer_tokens
    if budget <= 0:
        return ""

    entries_xml = []

    for r in results:
        score         = float(r.get("score", 0.0))
        model_id      = str(r.get("model_id", ""))
        miner_address = str(r.get("miner_address", ""))
        question_text = str(r.get("question_text", ""))
        thinking_text = str(r.get("thinking_text", "") or "")
        answer_text   = str(r.get("answer_text", "") or "")

        miner_short = miner_address[:10] if len(miner_address) >= 10 else miner_address

        # Estimate tokens for mandatory fields (everything except thinking)
        skeleton = (
            f"  <entry similarity=\"{score:.2f}\" model=\"{model_id}\" "
            f"miner=\"{miner_short}...\">\n"
            f"    <question>{question_text}</question>\n"
            f"    <thinking></thinking>\n"
            f"    <answer>{answer_text}</answer>\n"
            f"  </entry>\n"
        )
        skeleton_tokens = _estimate_tokens(skeleton)

        if skeleton_tokens > budget:
            # Even without thinking this entry doesn't fit — skip it
            continue

        # How many tokens remain for the thinking text?
        thinking_budget_tokens = budget - skeleton_tokens
        thinking_budget_chars  = thinking_budget_tokens * 4

        if len(thinking_text) > thinking_budget_chars:
            thinking_text = thinking_text[:max(0, thinking_budget_chars - 3)] + "..."

        entry = (
            f"  <entry similarity=\"{score:.2f}\" model=\"{model_id}\" "
            f"miner=\"{miner_short}...\">\n"
            f"    <question>{question_text}</question>\n"
            f"    <thinking>{thinking_text}</thinking>\n"
            f"    <answer>{answer_text}</answer>\n"
            f"  </entry>\n"
        )

        entry_tokens = _estimate_tokens(entry)
        budget -= entry_tokens
        entries_xml.append(entry)

        if budget <= 0:
            break

    if not entries_xml:
        return ""

    return OUTER_OPEN + "".join(entries_xml) + OUTER_CLOSE


# ── Prompt injection ──────────────────────────────────────────────────────────

def inject_context(prompt: str, prior_context: str) -> str:
    """
    Prepend prior_context to prompt, separated by a blank line.

    If prior_context is '' (empty string), returns prompt unchanged so
    inference is never disrupted by an absent memory store.
    """
    if not prior_context:
        return prompt
    return prior_context + "\n\n" + prompt
