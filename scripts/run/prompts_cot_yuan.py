"""Yuan et al. (2023) zero-shot CoT prompt templates.

Reference: Yuan, Xie, Huang, Ananiadou (2023),
"Zero-shot Temporal Relation Extraction with ChatGPT".
"""
from __future__ import annotations

import re


def mark_target_pair_in_doc(tokens, m1_tok_ids, m2_tok_ids, m1_id, m2_id):
    """Return doc text with ONLY the target pair wrapped in <EVENT eX>...</EVENT>.

    Non-target events appear as plain trigger words (no wrapper, no deletion).
    """
    marked = list(tokens)
    marked[m1_tok_ids[0]] = f"<EVENT e{m1_id}>{marked[m1_tok_ids[0]]}"
    marked[m1_tok_ids[-1]] = f"{marked[m1_tok_ids[-1]]}</EVENT>"
    marked[m2_tok_ids[0]] = f"<EVENT e{m2_id}>{marked[m2_tok_ids[0]]}"
    marked[m2_tok_ids[-1]] = f"{marked[m2_tok_ids[-1]]}</EVENT>"
    return " ".join(marked)


def ref(trigger, m_id):
    return f"<EVENT e{m_id}>{trigger}</EVENT>"


def q_same_event(doc_text, e1_ref, e2_ref):
    return (
        f"Given the following document:\n\n{doc_text}\n\n"
        f"Are {e1_ref} and {e2_ref} referring to the same event? "
        f"Keep the answer short and concise."
    )


def q_simultaneous_same_event(e1_ref, e2_ref):
    return (
        f"Did {e1_ref} and {e2_ref} simultaneously happen in that event? "
        f"Keep the answer short and concise."
    )


def q_simultaneous(e1_ref, e2_ref):
    return (
        f"Did {e1_ref} and {e2_ref} simultaneously happen? "
        f"Keep the answer short and concise."
    )


def q_before_same_event(e1_ref, e2_ref):
    return (
        f"Is {e1_ref} before {e2_ref} in that event? "
        f"Keep the answer short and concise."
    )


def q_after_same_event(e1_ref, e2_ref):
    return (
        f"Is {e1_ref} after {e2_ref} in that event? "
        f"Keep the answer short and concise."
    )


def q_before(e1_ref, e2_ref):
    return f"Is {e1_ref} before {e2_ref}? Keep the answer short and concise."


def q_after(e1_ref, e2_ref):
    return f"Is {e1_ref} after {e2_ref}? Keep the answer short and concise."


_UNCERTAIN_PATTERNS = [
    r"\bunknown\b", r"\bunclear\b", r"\bambiguous\b",
    r"\bnot specified\b", r"\bnot stated\b", r"\bnot provided\b",
    r"\bnot mentioned\b", r"\bnot possible\b", r"\bnot clear\b",
    r"\bcannot (?:be )?determine[d]?\b", r"\bcan't (?:be )?determine[d]?\b",
    r"\bcannot be inferred\b", r"\bimpossible to\b",
    r"\binsufficient\b", r"\bunsure\b", r"\bdifficult to (?:say|tell|determine)\b",
    r"\bi don'?t know\b", r"\bno information\b", r"\bnot enough (?:information|context)\b",
]


def parse_yes_no(response):
    """Return 'yes', 'no', or 'uncertain'. Strips <think>…</think> first."""
    if not response:
        return "uncertain"
    text = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip().lower()
    if not text:
        return "uncertain"

    first_sentence = re.split(r"[.\n!?]", text, maxsplit=1)[0]

    has_uncertainty = any(re.search(p, text) for p in _UNCERTAIN_PATTERNS)
    if has_uncertainty:
        if re.match(r"^\s*yes\b", first_sentence):
            return "yes"
        if re.match(r"^\s*no\b", first_sentence):
            return "no"
        return "uncertain"

    if re.match(r"^\s*yes\b", first_sentence):
        return "yes"
    if re.match(r"^\s*no\b", first_sentence):
        return "no"

    has_yes = bool(re.search(r"\byes\b", text))
    has_no = bool(re.search(r"\bno\b|\bnot\b", text))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "uncertain"
