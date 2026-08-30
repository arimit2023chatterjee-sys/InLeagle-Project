""" Builds the prompt that goes to the LLM (llama-3.1, in this case).
Two modes:
  legal : formal, cite exact sections and acts, professional tone
  layman : plain English, practical steps, no jargon

The mode is either chosen by the user explicitly
or auto-detected from the query.
"""

import re
import logging

log = logging.getLogger("prompt_builder")

# ─────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────

# Keywords that strongly signal layman mode
_LAYMAN_SIGNALS = [
    "explain", "what does", "what is", "what are", "in simple",
    "simple words", "layman", "easy", "i don't understand",
    "i dont understand", "what happens", "what should i do",
    "help me understand", "please explain", "can you explain",
    "what can i do", "what do i do", "i am confused",
    "scared", "worried", "help", "my bank", "they are threatening",
    "they called", "they came", "recovery agent", "they took",
]

# Keywords that strongly signal legal/formal mode
_LEGAL_SIGNALS = [
    "section", "act", "provision", "statute", "jurisdiction",
    "adjudicate", "tribunal", "precedent", "judgment", "citation",
    "legal position", "under law", "legal remedy", "relief",
    "petition", "writ", "pleading", "constitute", "liability",
    "pursuant", "thereof", "herein", "aforesaid",
]


def detect_mode(query: str) -> str:
    q = query.lower()
    layman_score = sum(1 for kw in _LAYMAN_SIGNALS if kw in q)
    legal_score  = sum(1 for kw in _LEGAL_SIGNALS  if kw in q)

    mode = "legal" if legal_score > layman_score else "layman"
    log.info(f"Mode detected: {mode}  (layman={layman_score}, legal={legal_score})")
    return mode


# ─────────────────────────────────────────────────────────────
# CONTEXT BUILDER
# Formats retrieved chunks into a readable numbered context block
# Eg: Each chunk is labelled with its source so the LLM can cite it.
    # Example output:
    # [1] SOURCE: SARFAESI Act 2002 (Act, 2002)
    #     TEXT: Section 13(2) — Where the borrower fails to discharge...

    # [2] SOURCE: Mardia Chemicals v Union of India 2004 (Supreme Court, 2004)
    #     TEXT: The Supreme Court held that while the Act is constitutional...
# ─────────────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant legal text found."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["document_name"]
        year   = chunk.get("year", "")
        stype  = chunk.get("source_type", "")
        court  = chunk.get("court", "")

        # Build a clean source label
        if stype == "case" and court:
            label = f"{source} ({court}, {year})"
        elif year:
            label = f"{source} ({year})"
        else:
            label = source

        lines.append(f"[{i}] SOURCE: {label}")
        lines.append(f"TEXT: {chunk['text'].strip()}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────

_LEGAL_SYSTEM_PROMPT = """You are an expert Indian banking law assistant with deep knowledge of all Indian banking legislation, RBI regulations, and landmark court judgments.

Your answers must follow these rules:
1. Base your answer primarily on the legal text provided in the context below. If the context is insufficient or the question is basic/general, you can supplement it using your general knowledge of Indian banking law, but clearly state what general legal provisions apply.
2. If the user greets you (e.g. "Hello", "Hi", "Who are you?"), respond professionally and explain what banking legislation you can help them analyze (SARFAESI, Section 138 NI Act, Banking Regulation Act, IBC, CIC, RBI circulars, etc.). Do not refuse to answer.
3. For every legal point derived from the context, cite the EXACT source — state the Act name, Section number, or Case name.
4. Use precise legal language.
5. Structure your answer clearly: legal position first, then relevant sections, then applicable cases if any."""

_LAYMAN_SYSTEM_PROMPT = """You are a friendly and knowledgeable Indian banking law guide named "Mr. InLeagle Singh", helping ordinary people understand their legal rights.

Your answers must follow these rules:
1. Base your answer primarily on the legal text provided in the context below. If the context does not contain the specific answer or if the query is a basic banking/legal question, feel free to use your general knowledge of Indian banking law to explain the concepts in simple terms, noting that it is general information.
2. If the user greets you or asks who you are, respond warmly and introduce yourself as "Mr. InLeagle Singh", explaining how you can help them navigate their banking and loan recovery problems.
3. Explain everything in plain, simple English — no legal jargon unless you immediately explain it.
4. Always mention which law or rule protects the person — but say it naturally (e.g. "Under the SARFAESI Act..." not "Pursuant to Section 13(2)...").
5. Give practical, actionable steps where possible.
6. Be empathetic — the person asking may be stressed or scared about their situation.
7. Never guess or fabricate legal provisions. Keep your answers accurate and helpful."""

_LEGAL_USER_TEMPLATE = """LEGAL CONTEXT:
{context}

QUESTION: {query}

Provide a precise legal answer citing the exact sections, acts, and cases from the context above."""

_LAYMAN_USER_TEMPLATE = """LEGAL CONTEXT:
{context}

QUESTION: {query}

Explain this in simple language a non-lawyer can understand, and tell me what steps I can take."""


# ─────────────────────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────────────────────

def build_prompt(
    query:  str,
    chunks: list[dict],
    mode:   str = "auto",  # "auto" | "legal" | "layman"
) -> tuple[str, str, str]:
    """
    Build the full prompt for the LLM.
    Returns:
        system_prompt  : the system instruction
        user_prompt    : the user message with context + query
        resolved_mode  : the mode that was actually used
    """
    # Resolve mode
    if mode == "auto":
        resolved_mode = detect_mode(query)
    else:
        resolved_mode = mode

    # Build context block from retrieved chunks
    context = build_context(chunks)

    # Select template
    if resolved_mode == "legal":
        system_prompt = _LEGAL_SYSTEM_PROMPT
        user_prompt   = _LEGAL_USER_TEMPLATE.format(
            context=context,
            query=query,
        )
    else:
        system_prompt = _LAYMAN_SYSTEM_PROMPT
        user_prompt   = _LAYMAN_USER_TEMPLATE.format(
            context=context,
            query=query,
        )

    return system_prompt, user_prompt, resolved_mode