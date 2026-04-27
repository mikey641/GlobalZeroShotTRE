"""One-shot probe: does the base R1-Distill-Qwen-14B produce <think> + label naturally?

Spins up a dedicated endpoint for the base model, runs 10 MATRES test pairs
through it, reports per-pair findings (has </think>? substantive chain?
parseable label? matches gold?), then DELETES the endpoint.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/eval/probe_base_r1distill.py
"""
from __future__ import annotations

import re
import sys
import time

from together import Together

from scripts.eval.eval_sft_student import build_prompt, load_test_pairs


BASE_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
N_PAIRS = 10
MAX_TOKENS = 4096
LABEL_RE = re.compile(r'\b(BEFORE|AFTER|EQUAL|VAGUE)\b')


def poll_until_started(c, endpoint_id, timeout_s=600):
    t0 = time.time()
    while True:
        e = c.endpoints.retrieve(endpoint_id)
        dt = int(time.time() - t0)
        print(f'  [+{dt:3}s] state={e.state}', flush=True)
        if e.state == 'STARTED':
            return e
        if e.state in ('ERROR', 'FAILED'):
            raise SystemExit(f'endpoint failed: {e}')
        if time.time() - t0 > timeout_s:
            raise SystemExit(f'endpoint did not start in {timeout_s}s')
        time.sleep(20)


def analyze(content):
    has_open  = '<think>' in content
    has_close = '</think>' in content
    # The template pre-fills `<think>\n`, so the model's *content* starts inside
    # think; presence of `</think>` is the real signal that a chain closed.
    chain = content.split('</think>', 1)[0] if has_close else content
    tail  = content.rsplit('</think>', 1)[-1] if has_close else ''
    chain_len = len(chain.strip())
    # Substantive = more than one line and mentions an EVENT tag from the prompt
    substantive = chain_len > 200 and '<EVENT' in chain
    m = LABEL_RE.search(tail.upper())
    label = m.group(1) if m else None
    return {
        'has_think_close': has_close,
        'has_think_open':  has_open,
        'chain_chars': chain_len,
        'substantive':     substantive,
        'label':           label,
    }


def main():
    c = Together()
    pairs = load_test_pairs()[:N_PAIRS]
    print(f'loaded {len(pairs)} pairs')

    print(f'\ncreating dedicated endpoint for {BASE_MODEL} ...')
    ep = c.endpoints.create(
        model=BASE_MODEL,
        display_name='probe-base-r1distill-14b',
        hardware='4x_nvidia_h100_80gb_sxm',
        autoscaling={'min_replicas': 1, 'max_replicas': 1},
        inactive_timeout=5,
        state='STARTED',
    )
    print(f'  id={ep.id}  name={ep.name}')
    try:
        poll_until_started(c, ep.id)
        ep_name = ep.name  # dedicated routing requires the hashed endpoint name

        print(f'\nrunning {N_PAIRS} probes via {ep_name}')
        rows = []
        for i, p in enumerate(pairs):
            prompt = build_prompt(p)
            t0 = time.time()
            r = c.chat.completions.create(
                model=ep_name,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            content = r.choices[0].message.content or ''
            info = analyze(content)
            info.update({
                'doc_id': p['doc_id'],
                'e1_id':  p['e1_id'],
                'e2_id':  p['e2_id'],
                'gold':   p['gold_label'],
                'tokens': r.usage.completion_tokens,
                'secs':   round(time.time() - t0, 1),
                'content': content,
            })
            rows.append(info)
            match = '✓' if info['label'] == p['gold_label'] else ('·' if info['label'] else '✗')
            print(f'  {i+1:2}/{N_PAIRS}  gold={p["gold_label"]:<6}  pred={str(info["label"]):<6}  '
                  f'{match}  close={int(info["has_think_close"])}  chain={info["chain_chars"]}  '
                  f'tok={info["tokens"]}  {info["secs"]}s')

        print('\n======== per-pair detail ========')
        for i, r in enumerate(rows):
            print(f'\n--- pair {i+1} ({r["doc_id"]}  {r["e1_id"]}-{r["e2_id"]})  '
                  f'gold={r["gold"]}  pred={r["label"]} ---')
            c1 = r['content']
            if len(c1) > 600:
                print(c1[:300])
                print(f'\n  ... [{len(c1) - 600} chars elided] ...\n')
                print(c1[-300:])
            else:
                print(c1)

        print('\n======== summary ========')
        n_close   = sum(1 for r in rows if r['has_think_close'])
        n_subst   = sum(1 for r in rows if r['substantive'])
        n_labeled = sum(1 for r in rows if r['label'])
        n_correct = sum(1 for r in rows if r['label'] == r['gold'])
        print(f'  pairs with </think> close:          {n_close}/{N_PAIRS}')
        print(f'  pairs with substantive chain:       {n_subst}/{N_PAIRS}')
        print(f'  pairs with parseable label:         {n_labeled}/{N_PAIRS}')
        print(f'  pairs where label matches gold:     {n_correct}/{N_PAIRS}')
    finally:
        print('\ndeleting endpoint ...')
        try:
            c.endpoints.delete(ep.id)
            print('  deleted')
        except Exception as e:
            print(f'  [warn] delete failed: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
