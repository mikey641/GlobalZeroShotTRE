#!/usr/bin/env python3
"""
v4e training worker — launched by torchrun from Modal.

Weighted SFT on DeepSeek-R1-Distill-Qwen-14B:
  user tokens   → weight 0   (masked)
  reasoning     → weight 1
  Yes/No commit → weight 10  (the single token before each EOS)

Uncertain turns: included as reasoning context but commit token masked.
"""

import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ── constants ──────────────────────────────────────────────────────────────────
MODEL_ID      = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATA_PATH     = os.environ.get("V4E_DATA_PATH",   "/data/train.jsonl")
OUTPUT_DIR    = os.environ.get("V4E_OUTPUT_DIR",  "/checkpoints/v4e")
DS_CONFIG     = os.environ.get("V4E_DS_CONFIG",   "/tmp/ds_config.json")
HF_REPO       = os.environ.get("V4E_HF_REPO",     "")

WEIGHT_REASON = 1.0
WEIGHT_COMMIT = 10.0
MAX_SEQ       = 4096   # p99 of training data is 3404
EPOCHS        = 2
LR            = 1e-5
BATCH_PER_GPU = 1
GRAD_ACC      = 8      # effective batch = 4 GPUs × 1 × 8 = 32

# DeepSeek-R1-Distill-Qwen-14B token IDs
EOS_ID       = 151643   # <｜end▁of▁sentence｜>
USER_ROLE_ID = 151644   # <｜User｜>
ASST_ROLE_ID = 151645   # <｜Assistant｜>
YES_ID       = 9454     # 'Yes'
NO_ID        = 2753     # 'No'

KEEP_RE = re.compile(r'\s*Keep the answer short and concise\.\s*$')


def strip_keep(s):
    return KEEP_RE.sub('', s).rstrip()


# Gold-correct yes/no for each turn position given the gold label.
# Turn 0=Q1 (same event?), 1=Q2 (simultaneous?), 2=Q3 (before?), 3=Q4 (after?)
_GOLD_ANSWER = {
    ('EQUAL',  0): 'yes',
    ('EQUAL',  1): 'yes',
    ('BEFORE', 0): 'no',  ('BEFORE', 1): 'no',  ('BEFORE', 2): 'yes',
    ('AFTER',  0): 'no',  ('AFTER',  1): 'no',  ('AFTER',  2): 'no',  ('AFTER',  3): 'yes',
    ('VAGUE',  0): 'no',  ('VAGUE',  1): 'no',  ('VAGUE',  2): 'no',  ('VAGUE',  3): 'no',
}

_EXPECTED_TURNS = {'EQUAL': 2, 'BEFORE': 3, 'AFTER': 4, 'VAGUE': 4}


def build_dataset(tok):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    with open(DATA_PATH) as f:
        raw = [json.loads(l) for l in f]

    records = []
    skipped = 0

    for r in raw:
        turns = r.get('turns') or []
        gold  = r.get('gold_label', '')
        if not turns or gold not in _EXPECTED_TURNS:
            skipped += 1
            continue

        if len(turns) != _EXPECTED_TURNS[gold]:
            skipped += 1
            continue

        # Reject rows where the teacher gave a certain but wrong answer.
        # Uncertain turns are restored using the gold-correct answer instead.
        bad = False
        for idx, t in enumerate(turns):
            p = t.get('parsed')
            if p in ('yes', 'no') and p != _GOLD_ANSWER.get((gold, idx), 'no'):
                bad = True
                break
        if bad:
            skipped += 1
            continue

        messages = []
        uncertain_flags = []
        for idx, t in enumerate(turns):
            q      = strip_keep(t.get('question', ''))
            th     = (t.get('think') or '').strip()
            p      = t.get('parsed')
            uncertain = (p == 'uncertain')
            ep     = _GOLD_ANSWER.get((gold, idx), 'no') if uncertain else p
            ans    = 'Yes' if ep == 'yes' else 'No'
            messages.append({'role': 'user',      'content': q})
            messages.append({'role': 'assistant', 'content': f'<think>\n{th}\n</Think>\n\n{ans}'})
            uncertain_flags.append(uncertain)

        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids  = tok.encode(text, add_special_tokens=False)

        if len(ids) > MAX_SEQ:
            skipped += 1
            continue

        # Find role boundaries (single-token markers)
        eos_positions  = [i for i, t in enumerate(ids) if t == EOS_ID]
        asst_positions = [i for i, t in enumerate(ids) if t == ASST_ROLE_ID]

        if len(eos_positions) != len(turns) or len(asst_positions) != len(turns):
            skipped += 1
            continue

        weights = [0.0] * len(ids)
        labels  = [-100]  * len(ids)

        for turn_idx in range(len(turns)):
            asst_start = asst_positions[turn_idx] + 1   # token after <｜Assistant｜>
            eos_pos    = eos_positions[turn_idx]

            # All assistant tokens (think block + commit + EOS): reasoning weight
            for i in range(asst_start, eos_pos + 1):
                labels[i]  = ids[i]
                weights[i] = WEIGHT_REASON

            # Commit token: upweight unless uncertain
            if not uncertain_flags[turn_idx]:
                prev = eos_pos - 1
                if prev >= asst_start and ids[prev] in (YES_ID, NO_ID):
                    weights[prev] = WEIGHT_COMMIT

        records.append({
            'input_ids':     ids,
            'attention_mask': [1] * len(ids),
            'labels':        labels,
            'token_weights': weights,
        })

    if local_rank == 0:
        print(f'Dataset: {len(records)} included, {skipped} skipped', flush=True)

    return Dataset.from_list(records)


def make_collator(tok):
    pad_id = tok.pad_token_id or EOS_ID

    def collate(batch):
        max_len = max(len(b['input_ids']) for b in batch)

        def pad(seq, val):
            return seq + [val] * (max_len - len(seq))

        return {
            'input_ids':      torch.tensor([pad(b['input_ids'],     pad_id) for b in batch]),
            'attention_mask': torch.tensor([pad(b['attention_mask'], 0)     for b in batch]),
            'labels':         torch.tensor([pad(b['labels'],         -100)  for b in batch]),
            'token_weights':  torch.tensor([pad(b['token_weights'],  0.0)   for b in batch],
                                           dtype=torch.float32),
        }

    return collate


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        token_weights = inputs.pop('token_weights')   # (B, L)
        labels        = inputs.pop('labels')          # (B, L)

        outputs = model(**inputs)
        logits  = outputs.logits                      # (B, L, V)

        # Causal shift: logit at position i predicts label at position i+1
        shift_logits  = logits[:, :-1].contiguous()        # (B, L-1, V)
        shift_labels  = labels[:, 1:].contiguous()         # (B, L-1)
        shift_weights = token_weights[:, 1:].contiguous()  # (B, L-1)

        loss_fct  = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())                        # (B, L-1)

        active    = (shift_labels != -100)
        loss      = (per_token * shift_weights * active).sum() / \
                    (shift_weights * active).sum().clamp(min=1.0)

        return (loss, outputs) if return_outputs else loss


def write_ds_config():
    config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param":     {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": BATCH_PER_GPU,
        "gradient_accumulation_steps": GRAD_ACC,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }
    with open(DS_CONFIG, 'w') as f:
        json.dump(config, f)


def main():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    write_ds_config()

    if local_rank == 0:
        print('Loading tokenizer...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = 'right'

    ds = build_dataset(tok)

    if local_rank == 0:
        print('Loading model...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    )
    model.config.use_cache = False

    run_name = 'matres-v4e-14b-weighted'
    if local_rank == 0:
        import wandb
        wandb.init(
            project='tre-matres',
            name=run_name,
        )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_PER_GPU,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=1,
        deepspeed=DS_CONFIG,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        remove_unused_columns=False,
        report_to='wandb' if local_rank == 0 else 'none',
        run_name=run_name,
        seed=42,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=make_collator(tok),
        tokenizer=tok,
    )

    if local_rank == 0:
        print('Training...', flush=True)
    trainer.train()

    if local_rank == 0:
        trainer.save_model(OUTPUT_DIR)
        tok.save_pretrained(OUTPUT_DIR)
        print(f'Saved to {OUTPUT_DIR}', flush=True)

        if HF_REPO:
            hf_token = os.environ.get('HF_TOKEN', '')
            if hf_token:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                api.create_repo(HF_REPO, exist_ok=True, private=True)
                api.upload_folder(folder_path=OUTPUT_DIR, repo_id=HF_REPO)
                print(f'Pushed to {HF_REPO}', flush=True)
            else:
                print('HF_TOKEN not set — skipping HF Hub push', flush=True)


if __name__ == '__main__':
    main()
