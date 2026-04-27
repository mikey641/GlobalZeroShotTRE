"""Smoke-test v3 epoch-2 LoRA checkpoint on 10 MATRES test pairs.

Creates a dedicated endpoint (4xH100-80GB SXM) for
  mikey641_af35/DeepSeek-R1-Distill-Qwen-14B-tre-elim-v3-1adc8f34-step-56
routes chat via the endpoint's hashed `name`, runs 10 probes, prints per-pair
detail + summary. Does NOT shut down the endpoint.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/eval/smoke_v3_epoch2.py
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from together import Together

from scripts.eval.eval_sft_student import (
    build_prompt, load_test_pairs, parse_label,
)


MODEL = 'mikey641_af35/DeepSeek-R1-Distill-Qwen-14B-tre-elim-v3-1adc8f34-step-56'
DISPLAY = 'tre-elim-v3-epoch2-step56'
N_PAIRS = 10
MAX_TOKENS = 8192
MAX_WORKERS = 2


def poll_until_started(c, endpoint_id, timeout_s=900):
    t0 = time.time()
    while True:
        e = c.endpoints.retrieve(endpoint_id)
        dt = int(time.time() - t0)
        print(f'  [+{dt:4}s] state={e.state}', flush=True)
        if e.state == 'STARTED':
            return e
        if e.state in ('ERROR', 'FAILED'):
            raise SystemExit(f'endpoint failed: {e}')
        if time.time() - t0 > timeout_s:
            raise SystemExit(f'endpoint did not start in {timeout_s}s')
        time.sleep(20)


def analyze(content):
    has_cap   = '</Think>' in content
    has_low   = '</think>' in content
    # Split on whichever close tag appears, prefer capital-T (expected from v3)
    if has_cap:
        chain = content.split('</Think>', 1)[0]
        tail  = content.rsplit('</Think>', 1)[-1]
    elif has_low:
        chain = content.split('</think>', 1)[0]
        tail  = content.rsplit('</think>', 1)[-1]
    else:
        chain, tail = content, ''
    chain_len = len(chain.strip())
    substantive = chain_len > 200 and '<EVENT' in chain
    return {
        'has_think_close_cap': has_cap,
        'has_think_close_low': has_low,
        'chain_chars': chain_len,
        'substantive': substantive,
        'label': parse_label(content),
    }


def main():
    c = Together()
    pairs = load_test_pairs()[:N_PAIRS]
    print(f'loaded {len(pairs)} pairs')

    print(f'\ncreating dedicated endpoint for {MODEL} ...', flush=True)
    ep = c.endpoints.create(
        model=MODEL,
        display_name=DISPLAY,
        hardware='4x_nvidia_h100_80gb_sxm',
        autoscaling={'min_replicas': 1, 'max_replicas': 1},
        inactive_timeout=15,
        state='STARTED',
    )
    print(f'  id={ep.id}')
    print(f'  name={ep.name}')
    print(f'  state={ep.state}')

    poll_until_started(c, ep.id)
    ep_name = ep.name

    print(f'\nrunning {N_PAIRS} probes via {ep_name}', flush=True)

    def work(i, p):
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
            'i':      i,
            'doc_id': p['doc_id'],
            'e1_id':  p['e1_id'],
            'e2_id':  p['e2_id'],
            'gold':   p['gold_label'],
            'tokens': r.usage.completion_tokens,
            'secs':   round(time.time() - t0, 1),
            'content': content,
        })
        return info

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(work, i, p): (i, p) for i, p in enumerate(pairs, 1)}
        for fut in as_completed(futs):
            try:
                info = fut.result()
            except Exception as e:
                i, p = futs[fut]
                print(f'  pair {i}: ERROR {type(e).__name__}: {e}', file=sys.stderr)
                continue
            rows.append(info)
            m = '✓' if info['label'] == info['gold'] else ('·' if info['label'] else '✗')
            print(f'  {info["i"]:2}/{N_PAIRS}  gold={info["gold"]:<6}  pred={str(info["label"]):<6}  '
                  f'{m}  closeT={int(info["has_think_close_cap"])}  chain={info["chain_chars"]}  '
                  f'tok={info["tokens"]}  {info["secs"]}s', flush=True)

    rows.sort(key=lambda r: r['i'])

    print('\n======== per-pair detail ========')
    for r in rows:
        print(f'\n--- pair {r["i"]} ({r["doc_id"]}  {r["e1_id"]}-{r["e2_id"]})  '
              f'gold={r["gold"]}  pred={r["label"]} ---')
        c1 = r['content']
        if len(c1) > 1200:
            print(c1[:600])
            print(f'\n  ... [{len(c1) - 1200} chars elided] ...\n')
            print(c1[-600:])
        else:
            print(c1)

    print('\n======== summary ========')
    n_cap     = sum(1 for r in rows if r['has_think_close_cap'])
    n_low     = sum(1 for r in rows if r['has_think_close_low'])
    n_subst   = sum(1 for r in rows if r['substantive'])
    n_labeled = sum(1 for r in rows if r['label'])
    n_correct = sum(1 for r in rows if r['label'] == r['gold'])
    from collections import Counter
    pred_dist = Counter(r['label'] for r in rows)
    gold_dist = Counter(r['gold'] for r in rows)
    print(f'  pairs with </Think> close (cap):    {n_cap}/{len(rows)}')
    print(f'  pairs with </think> close (low):    {n_low}/{len(rows)}')
    print(f'  pairs with substantive chain:       {n_subst}/{len(rows)}')
    print(f'  pairs with parseable label:         {n_labeled}/{len(rows)}')
    print(f'  pairs where label matches gold:     {n_correct}/{len(rows)}')
    print(f'  pred dist: {dict(pred_dist)}')
    print(f'  gold dist: {dict(gold_dist)}')

    print(f'\nENDPOINT LEFT RUNNING: id={ep.id}  name={ep_name}')
    print(f'  to stop later: Together().endpoints.delete("{ep.id}")')


if __name__ == '__main__':
    main()
