"""
Modal launcher for v4e — weighted SFT of DeepSeek-R1-Distill-Qwen-14B.

One-time setup:
    # Upload training data to Modal volume (only needed once / when data changes):
    modal volume put matres-train-data \\
        GlobalZeroShotTRE/output/matres_train_v4d_full_reasoning_goldcond.jsonl \\
        train.jsonl

    # Upload MATRES test dir for eval:
    modal volume put matres-train-data GlobalZeroShotTRE/data/MATRES MATRES

Usage:
    # Cost estimate (no GPU, runs locally):
    modal run GlobalZeroShotTRE/scripts/train/modal_v4e.py

    # Launch training:
    modal run GlobalZeroShotTRE/scripts/train/modal_v4e.py::train

    # Run eval inside Modal (after training):
    modal run GlobalZeroShotTRE/scripts/train/modal_v4e.py::eval_matres
"""

import os
from pathlib import Path

import modal

_HERE  = Path(__file__).parent
ROOT   = _HERE.parent.parent.parent    # self-improving-tre/
ENV    = ROOT / ".env"
WORKER = _HERE / "train_worker_v4e.py"

DATA_PATH   = "/data/train.jsonl"
MATRES_DIR  = "/data/MATRES"
CKPT_DIR    = "/checkpoints/v4e"
HF_REPO     = ""   # e.g. "mikey641/matres-v4e-14b"; leave empty to skip HF push

app = modal.App("matres-v4e")

# ── Training image ─────────────────────────────────────────────────────────────
# Worker script is baked in at image-build time (tiny file, fine to cache).
train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.44.0,<5.0",
        "accelerate>=0.30.0",
        "deepspeed>=0.14.4",
        "datasets>=2.20.0",
        "wandb",
        "huggingface_hub>=0.23.0",
    )
    .env({"HF_HOME": "/hf_cache", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_file(str(WORKER), "/worker/train_worker_v4e.py")
)

# ── Inference image ────────────────────────────────────────────────────────────
# Source code is added as a separate thin layer on top of the cached vllm base,
# so the heavy vllm install is only done once even when source changes.
_vllm_base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.7.3", "transformers>=4.44.0,<5.0",
                 "huggingface_hub>=0.23.0", "together", "numpy", "scipy",
                 "scikit-learn")
    .env({"HF_HOME": "/hf_cache"})
)
_MATRES_TEST = ROOT / "GlobalZeroShotTRE" / "data" / "MATRES" / "_in_OmniTemp_format" / "test"
vllm_image = (
    _vllm_base
    .add_local_dir(
        str(ROOT / "GlobalZeroShotTRE"), "/app",
        copy=True,
        ignore=[".git", "**/.git/**", "output", "**/__pycache__", "*.pyc", "*.parquet", "*.jsonl"],
    )
    .add_local_dir(
        str(_MATRES_TEST), "/app/data/MATRES/_in_OmniTemp_format/test",
        copy=True,
    )
)

# ── Volumes ────────────────────────────────────────────────────────────────────
data_vol = modal.Volume.from_name("matres-train-data",  create_if_missing=True)
ckpt_vol = modal.Volume.from_name("matres-checkpoints", create_if_missing=True)
hf_vol   = modal.Volume.from_name("hf-model-cache",    create_if_missing=True)


# ── Training ───────────────────────────────────────────────────────────────────
@app.function(
    image=train_image,
    gpu="H100:4",
    timeout=5 * 3600,
    volumes={
        "/data":        data_vol,
        "/checkpoints": ckpt_vol,
        "/hf_cache":    hf_vol,
    },
    secrets=[modal.Secret.from_dotenv(str(ENV))],
)
def train():
    import subprocess
    subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=4",
            "--master_port=29500",
            "/worker/train_worker_v4e.py",
        ],
        env={
            **os.environ,
            "V4E_DATA_PATH":  DATA_PATH,
            "V4E_OUTPUT_DIR": CKPT_DIR,
            "V4E_DS_CONFIG":  "/tmp/ds_config.json",
            "V4E_HF_REPO":    HF_REPO,
        },
        check=True,
    )
    ckpt_vol.commit()
    print("Training complete — checkpoint committed to volume.", flush=True)


# ── vLLM serving (OpenAI-compatible chat completions) ─────────────────────────
@app.cls(
    image=vllm_image,
    gpu="H100:1",
    timeout=2 * 3600,
    volumes={
        "/checkpoints": ckpt_vol,
        "/hf_cache":    hf_vol,
    },
)
class ModelServer:
    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams
        self.llm    = LLM(model=CKPT_DIR, dtype="bfloat16", max_model_len=8192,
                          trust_remote_code=True)
        self.SP     = SamplingParams
        from transformers import AutoTokenizer
        self.tok    = AutoTokenizer.from_pretrained(CKPT_DIR)

    @modal.method()
    def generate(self, prompts: list[str], max_tokens: int = 4096,
                 temperature: float = 0.0) -> list[str]:
        params   = self.SP(max_tokens=max_tokens, temperature=temperature)
        outputs  = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    @modal.fastapi_endpoint(method="POST")
    def v1_chat_completions(self, request: dict) -> dict:
        """Minimal OpenAI-compatible /v1/chat/completions endpoint."""
        msgs = request.get("messages", [])
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        result = self.generate.local([prompt],
                                     max_tokens=request.get("max_tokens", 4096),
                                     temperature=request.get("temperature", 0.0))
        return {
            "choices": [{"message": {"role": "assistant", "content": result[0]},
                         "finish_reason": "stop"}],
            "model": CKPT_DIR,
        }


# ── Checkpoint commit ─────────────────────────────────────────────────────────
@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    volumes={"/checkpoints": ckpt_vol},
    timeout=120,
)
def commit_checkpoints():
    """Commit the checkpoint volume so eval can see the latest files."""
    import os
    files = []
    for root, dirs, fnames in os.walk("/checkpoints"):
        for fn in fnames:
            files.append(os.path.join(root, fn))
    print(f"Found {len(files)} files in /checkpoints")
    if files:
        print("Sample:", files[:5])
    ckpt_vol.commit()
    print("Checkpoint volume committed.")


# ── Eval inside Modal ──────────────────────────────────────────────────────────
@app.function(
    image=vllm_image,
    gpu="H100:1",
    timeout=3 * 3600,
    volumes={
        "/data":        data_vol,
        "/checkpoints": ckpt_vol,
        "/hf_cache":    hf_vol,
    },
)
def eval_matres():
    """Run MATRES test eval on v4e checkpoint via vLLM."""
    import json, re, sys
    from pathlib import Path
    sys.path.insert(0, '/app')

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Trainer saves to checkpoint-N subdirs; find the latest one if top-level is absent
    ckpt = CKPT_DIR
    ckpt_path = Path(CKPT_DIR)
    if not (ckpt_path / "config.json").exists():
        subdirs = sorted(ckpt_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if subdirs:
            ckpt = str(subdirs[-1])
            print(f"Using checkpoint: {ckpt}", flush=True)

    print("Loading v4e checkpoint...", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    llm = LLM(model=ckpt, dtype="bfloat16", max_model_len=8192, trust_remote_code=True,
              enforce_eager=True)
    sp  = SamplingParams(max_tokens=4096, temperature=0.0)

    import scripts.eval.eval_sft_student as _eval_sft
    _eval_sft.TEST_FOLDER = '/app/data/MATRES/_in_OmniTemp_format/test'
    from scripts.eval.eval_sft_student import load_test_pairs
    from scripts.run.prompts_cot_yuan  import mark_target_pair_in_doc, ref

    pairs = load_test_pairs()
    print(f"Loaded {len(pairs)} test pairs", flush=True)

    # Precompute marked doc text and event refs for each pair
    pair_meta = []
    for pair in pairs:
        tokens, ment = pair['_ctx']
        m1, m2 = ment[pair['e1_id']], ment[pair['e2_id']]
        marked = mark_target_pair_in_doc(tokens, m1['tokens_ids'], m2['tokens_ids'],
                                         m1['m_id'], m2['m_id'])
        e1r = ref(m1['tokens'], m1['m_id'])
        e2r = ref(m2['tokens'], m2['m_id'])
        pair_meta.append((marked, e1r, e2r))

    YESNO_RE = re.compile(r'\b(yes|no)\b', re.IGNORECASE)

    def parse_yesno(text):
        tail = text
        for tag in ('</Think>', '</think>'):
            if tag in tail:
                tail = tail.rsplit(tag, 1)[-1]
                break
        m = YESNO_RE.search(tail.strip())
        return m.group(1).lower() if m else None

    def batch_call(history_list):
        """Run one chain step for a batch of conversations simultaneously."""
        prompts = [
            tok.apply_chat_template(h, tokenize=False, add_generation_prompt=True)
            for h in history_list
        ]
        outputs = llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outputs]

    def q_text(qid, e1r, e2r, in_event):
        sfx = ' in that event' if in_event else ''
        if qid == 'Q2': return f"Did {e1r} and {e2r} simultaneously happen{sfx}?"
        if qid == 'Q3': return f"Is {e1r} before {e2r}{sfx}?"
        if qid == 'Q4': return f"Is {e1r} after {e2r}{sfx}?"

    # Batch the chain: 4 rounds of vLLM batch generation over all 837 pairs.
    # Round 1: Q1 (same-event check) — all pairs in one batch
    histories = []
    for pair, (marked, e1r, e2r) in zip(pairs, pair_meta):
        q1 = (f"Given the following document:\n\n{marked}\n\n"
              f"Are {e1r} and {e2r} referring to the same event?")
        histories.append([{'role': 'user', 'content': q1}])

    print("Round Q1...", flush=True)
    r1s = batch_call(histories)
    in_events = []
    for i, r1 in enumerate(r1s):
        histories[i].append({'role': 'assistant', 'content': r1})
        in_events.append(parse_yesno(r1) == 'yes')

    # Rounds Q2–Q4: each builds on prior history
    preds = {}
    done  = [False] * len(pairs)

    for qid in ('Q2', 'Q3', 'Q4'):
        active_idx = [i for i, d in enumerate(done) if not d]
        print(f"Round {qid}: {len(active_idx)} active pairs...", flush=True)

        batch_hist = []
        for i in active_idx:
            _, e1r, e2r = pair_meta[i]
            q = q_text(qid, e1r, e2r, in_events[i])
            histories[i].append({'role': 'user', 'content': q})
            batch_hist.append(histories[i])

        responses = batch_call(batch_hist)
        for j, i in enumerate(active_idx):
            r   = responses[j]
            histories[i].append({'role': 'assistant', 'content': r})
            ans = parse_yesno(r)
            pair = pairs[i]
            key  = (pair['doc_id'], pair['e1_id'], pair['e2_id'])
            if qid == 'Q2' and ans == 'yes':
                preds[key] = 'EQUAL';  done[i] = True
            elif qid == 'Q3' and ans == 'yes':
                preds[key] = 'BEFORE'; done[i] = True
            elif qid == 'Q4':
                preds[key] = 'AFTER' if ans == 'yes' else 'VAGUE'
                done[i] = True

    # Any pair that never terminated (shouldn't happen) defaults to VAGUE
    for i, pair in enumerate(pairs):
        key = (pair['doc_id'], pair['e1_id'], pair['e2_id'])
        if key not in preds:
            preds[key] = 'VAGUE'

    from scripts.eval.shared.evaluation import evaluation
    from scripts.utils.classes.datasets_type import MatresDataset
    from scripts.utils.classes.label_sets import FourRelsLabels

    label_set = FourRelsLabels()
    golds_list, preds_list = [], []
    for pair in pairs:
        key = (pair['doc_id'], pair['e1_id'], pair['e2_id'])
        golds_list.append(label_set[pair['gold_label']])
        preds_list.append(label_set[preds.get(key, 'VAGUE')])

    f1 = evaluation(golds_list, preds_list, None, None, MatresDataset())
    print(f"\n{'='*40}")
    print(f"v4e MATRES F1: {f1:.4f}")
    print(f"{'='*40}")
    return {'f1': f1}


# ── Cost estimate (default local entrypoint) ───────────────────────────────────
@app.local_entrypoint()
def main():
    rows        = 11_883
    avg_tok     = 1_796
    tok_2ep     = rows * avg_tok * 2        # 2 epochs worth of tokens
    throughput  = 16_000                    # tok/s — 4×H100, ZeRO-3, 14B (conservative)
    train_sec   = tok_2ep / throughput
    setup_sec   = 15 * 60                   # image pull + HF model download

    h100_rate   = 5.40                      # USD/GPU/hr — verify at modal.com/pricing
    train_cost  = (train_sec / 3600) * 4 * h100_rate
    setup_cost  = (setup_sec / 3600) * 4 * h100_rate

    # 4 batched rounds of 837 pairs; vLLM batches all simultaneously (~3 min/round)
    eval_sec    = 4 * 3 * 60
    eval_cost   = (eval_sec / 3600) * 1 * h100_rate

    total       = train_cost + setup_cost + eval_cost

    print(f"""
╔══════════════════════════════════════════════════════╗
║          v4e Modal cost estimate                     ║
╠══════════════════════════════════════════════════════╣
║  Model     DeepSeek-R1-Distill-Qwen-14B              ║
║  Hardware  4×H100-80GB, DeepSpeed ZeRO-3, full FT   ║
║  Data      {rows:,} rows · ~{tok_2ep/1e6:.0f}M tokens (2 epochs)       ║
╠══════════════════════════════════════════════════════╣
║  Training time    {train_sec/60:.0f} min                        ║
║  Setup overhead   {setup_sec//60} min (image + model download)  ║
║  Eval time        {eval_sec/60:.0f} min (1×H100, serial turns)  ║
╠══════════════════════════════════════════════════════╣
║  H100 rate (spot) ~${h100_rate:.2f}/GPU/hr  ←  verify!       ║
║  Training cost    ${train_cost:.0f}                           ║
║  Setup cost       ${setup_cost:.0f}                           ║
║  Eval cost        ${eval_cost:.2f}                          ║
║  ─────────────────────────────────────────────────  ║
║  TOTAL ESTIMATE   ~${total:.0f}  (±30%)                   ║
╚══════════════════════════════════════════════════════╝

Next steps:
  1. Upload training data (once):
       modal volume put matres-train-data \\
         GlobalZeroShotTRE/output/matres_train_v4d_full_reasoning_goldcond.jsonl \\
         train.jsonl

  2. Launch training:
       modal run GlobalZeroShotTRE/scripts/train/modal_v4e.py::train

  3. Run eval:
       modal run GlobalZeroShotTRE/scripts/train/modal_v4e.py::eval_matres
""")
