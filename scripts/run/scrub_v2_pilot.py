"""Post-process v2 output: drop forced pairs (F1) + drop hint-conflict pairs (F3)
+ scrub `\\bhint\\b` sentences from gold-conditioned <think> blocks (F2).
No new R1 calls.

Default paths target the pilot. Override via --input / --unscrubbed_out /
--scrubbed_out for the full run.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "output" / "matres_train_continue_r1_goldconditioned_v2_pilot50.jsonl"
DEFAULT_DST_UNSCRUBBED = ROOT / "output" / "matres_train_continue_r1_goldconditioned_v2_pilot50_unscrubbed.jsonl"
DEFAULT_DST_SCRUBBED = ROOT / "output" / "matres_train_continue_r1_goldconditioned_v2_pilot50_scrubbed.jsonl"

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"*])|\n\n+')
HINT_RE = re.compile(r'\bhint\b', re.IGNORECASE)

# Filter 3: hint-conflict patterns on UNSCRUBBED <think> of gold-conditioned turns.
# If any of these match, the trace shows internal disagreement that may produce
# chain-label inconsistency post-scrub.
HINT_CONFLICT_RE = re.compile(
    r"\bdespite the hint\b"
    r"|\bregardless of (the )?hint\b"
    r"|\bcontrary to (the )?hint\b"
    r"|\bhint (might be|may be|is|could be) (wrong|incorrect|a trick|misleading)\b"
    r"|\bhint .{0,40} contradicts?\b"
    r"|\bcomply (with|even if) .{0,40}\b.{0,80}\bhint\b"
    r"|\bas per the hint\b",
    re.IGNORECASE | re.DOTALL,
)


def split_sentences(text):
    if not text:
        return []
    parts = SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def scrub_think(think):
    """Drop sentences containing \\bhint\\b. Return (cleaned_text, n_dropped, n_total)."""
    sents = split_sentences(think)
    kept = [s for s in sents if not HINT_RE.search(s)]
    return " ".join(kept), len(sents) - len(kept), len(sents)


def reconstruct_response(think, original_response):
    """Rebuild the response field with the cleaned think.
    Original format: '<think>...</think>\\n\\n<answer>'.
    """
    m = re.match(r'<think>(.*?)</think>\s*(.*)', original_response, re.DOTALL)
    if not m:
        return original_response  # nothing to do
    answer_part = m.group(2)
    return f"<think>{think}</think>\n\n{answer_part}"


def main(src=DEFAULT_SRC, dst_unscrubbed=DEFAULT_DST_UNSCRUBBED, dst_scrubbed=DEFAULT_DST_SCRUBBED):
    src = Path(src)
    dst_unscrubbed = Path(dst_unscrubbed)
    dst_scrubbed = Path(dst_scrubbed)
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} rows from {src}")

    # ---- Filter 1: drop forced-commit pairs ----
    forced_rows = [r for r in rows if r.get("forced_commits")]
    after_f1 = [r for r in rows if not r.get("forced_commits")]
    print(f"[F1] Dropped {len(forced_rows)} rows with any forced commit. Remaining {len(after_f1)}.")

    # ---- Filter 3: drop pairs whose UNSCRUBBED <think> shows hint-conflict ----
    # Scan only gold-conditioned turns (idx >= divergence_turn).
    hint_conflict_rows = []
    after_f3 = []
    for r in after_f1:
        dv = r["divergence_turn"]
        matched = []
        for j, t in enumerate(r["turns"]):
            if j >= dv:
                m = HINT_CONFLICT_RE.search(t.get("think", "") or "")
                if m:
                    matched.append({"turn_idx": j, "match": m.group(0)})
        if matched:
            r2 = dict(r)
            r2["_hint_conflict_matches"] = matched
            hint_conflict_rows.append(r2)
        else:
            after_f3.append(r)
    print(f"[F3] Dropped {len(hint_conflict_rows)} rows with hint-conflict reasoning. "
          f"Remaining {len(after_f3)}.")

    kept_rows = after_f3

    # Save unscrubbed (after F1 + F3, <think> untouched) for transparency
    with dst_unscrubbed.open("w") as f:
        for r in kept_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote unscrubbed → {dst_unscrubbed}")

    # ---- Filter 2: scrub hint sentences from gold-conditioned <think> blocks ----
    # Track stats
    pre_lengths = []
    post_lengths = []
    n_sents_dropped_total = 0
    n_sents_total = 0
    n_short_after = 0  # post-scrub <think> < 100 chars
    short_examples = []
    rows_touched = 0

    scrubbed_rows = []
    for r in kept_rows:
        dv = r["divergence_turn"]
        new_turns = []
        row_touched = False
        for j, t in enumerate(r["turns"]):
            if j >= dv:
                pre = t["think"] or ""
                cleaned, n_drop, n_tot = scrub_think(pre)
                pre_lengths.append(len(pre))
                post_lengths.append(len(cleaned))
                n_sents_dropped_total += n_drop
                n_sents_total += n_tot
                if n_drop > 0:
                    row_touched = True
                if len(cleaned) < 100:
                    n_short_after += 1
                    if len(short_examples) < 5:
                        short_examples.append({
                            "doc_id": r["doc_id"],
                            "turn_idx": j,
                            "pre_len": len(pre),
                            "post_len": len(cleaned),
                            "post_text": cleaned,
                        })
                new_t = dict(t)
                new_t["think"] = cleaned
                new_t["response"] = reconstruct_response(cleaned, t["response"])
                new_turns.append(new_t)
            else:
                new_turns.append(t)
        if row_touched:
            rows_touched += 1
        new_r = dict(r)
        new_r["turns"] = new_turns
        scrubbed_rows.append(new_r)

    with dst_scrubbed.open("w") as f:
        for r in scrubbed_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote scrubbed → {dst_scrubbed}")

    # ---- Verification ----
    # 1. No <think> in saved rows should contain \bhint\b (across all turns)
    leak_count = 0
    for r in scrubbed_rows:
        for t in r["turns"]:
            if HINT_RE.search(t.get("think", "") or ""):
                leak_count += 1
    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Hint mentions remaining in saved <think> blocks: {leak_count}  (should be 0)")
    print()

    # 2. <think> length distribution (gold-conditioned turns only)
    def stats(xs):
        if not xs:
            return None
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        return {
            "n": n,
            "median": statistics.median(xs_sorted),
            "p10": xs_sorted[max(0, n // 10)],
            "p90": xs_sorted[min(n - 1, (9 * n) // 10)],
            "min": xs_sorted[0],
            "max": xs_sorted[-1],
        }

    print(f"Gold-conditioned <think> length stats:")
    print(f"  pre-scrub:  {stats(pre_lengths)}")
    print(f"  post-scrub: {stats(post_lengths)}")
    print()
    print(f"Gold-conditioned turns total: {len(post_lengths)}")
    print(f"Sentences dropped: {n_sents_dropped_total} / {n_sents_total} "
          f"({100*n_sents_dropped_total/max(1,n_sents_total):.1f}%)")
    print(f"Rows where at least one sentence was scrubbed: {rows_touched}/{len(kept_rows)}")
    print(f"Post-scrub <think> blocks under 100 chars: {n_short_after}")
    if short_examples:
        print()
        print("Short post-scrub examples (first 5):")
        for ex in short_examples:
            print(f"  doc={ex['doc_id']} turn_idx={ex['turn_idx']} "
                  f"pre={ex['pre_len']}→post={ex['post_len']}")
            print(f"    {ex['post_text']!r}")

    # ---- Final summary ----
    print()
    print("=" * 60)
    print("FINAL")
    print("=" * 60)
    print(f"Original input:        {len(rows)}")
    print(f"Forced-drops (F1):     {len(forced_rows)}")
    print(f"Hint-conflict drops:   {len(hint_conflict_rows)}")
    print(f"Final saved:           {len(scrubbed_rows)}")

    # Per-class breakdowns
    from collections import Counter
    print()
    print(f"Final saved label distribution: {dict(Counter(r['gold_label'] for r in scrubbed_rows))}")
    print(f"F1 dropped label distribution:  {dict(Counter(r['gold_label'] for r in forced_rows))}")
    print(f"F3 dropped label distribution:  {dict(Counter(r['gold_label'] for r in hint_conflict_rows))}")

    # Print all drops only if small enough; otherwise just first 10 each
    print()
    print(f"Forced-dropped rows (F1) [showing up to 20]:")
    for fr in forced_rows[:20]:
        print(f"  doc={fr['doc_id']} pair=({fr['e1_id']},{fr['e2_id']}) "
              f"gold={fr['gold_label']} forced_turns={fr['forced_commits']}")
    if len(forced_rows) > 20:
        print(f"  ... and {len(forced_rows) - 20} more")

    print()
    print(f"Hint-conflict dropped rows (F3) [showing up to 20]:")
    for fr in hint_conflict_rows[:20]:
        matches = fr["_hint_conflict_matches"]
        match_summary = "; ".join(f"t{m['turn_idx']}={m['match']!r}" for m in matches[:3])
        print(f"  doc={fr['doc_id']} pair=({fr['e1_id']},{fr['e2_id']}) "
              f"gold={fr['gold_label']} matches=[{match_summary}]")
    if len(hint_conflict_rows) > 20:
        print(f"  ... and {len(hint_conflict_rows) - 20} more")

    # Overlap check: any row in both?
    forced_keys = {(r['doc_id'], r['e1_id'], r['e2_id']) for r in forced_rows}
    conflict_keys = {(r['doc_id'], r['e1_id'], r['e2_id']) for r in hint_conflict_rows}
    overlap = forced_keys & conflict_keys
    print()
    print(f"Overlap (rows in both F1 and F3): {len(overlap)}  "
          f"(N/A — F3 only sees rows that survived F1)")

    # Specific verification
    target = ("NYT20000330.0406.json", "49", "53")
    print()
    in_conflict = target in conflict_keys
    in_final = any((r['doc_id'], r['e1_id'], r['e2_id']) == target for r in scrubbed_rows)
    print(f"Verify NYT20000330.0406 (49,53):")
    print(f"  in hint-conflict drops: {in_conflict}")
    print(f"  in final saved set:     {in_final}  (should be False)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DEFAULT_SRC))
    p.add_argument("--unscrubbed_out", default=None,
                   help="default: <input_stem>_unscrubbed.jsonl")
    p.add_argument("--scrubbed_out", default=None,
                   help="default: <input_stem>_scrubbed.jsonl")
    args = p.parse_args()
    src = Path(args.input)
    unscrubbed = Path(args.unscrubbed_out) if args.unscrubbed_out else src.with_name(src.stem + "_unscrubbed.jsonl")
    scrubbed = Path(args.scrubbed_out) if args.scrubbed_out else src.with_name(src.stem + "_scrubbed.jsonl")
    main(src, unscrubbed, scrubbed)
