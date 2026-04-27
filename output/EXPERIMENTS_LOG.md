# MATRES SFT Experiments Ledger

Last updated: 2026-04-25 session.

## Versions at a glance

| version | format | rows | size | base | training | F1 | acc | notes |
|---|---|---|---|---|---|---|---|---|
| **v2** | `<think>` lowercase + label, single-turn | ~7,729 | 64 MB | R1-Distill-14B | LoRA r=64 | 0.3459 | — | template-strip wiped reasoning; collapsed to AFTER for 835/837 preds |
| **v3** | `<think>...</Think>` capital-T + label, single-turn | 7,729 | 64 MB | R1-Distill-14B | LoRA r=64, 3 epochs | 0.6058 (eval) / 0.6378 (oracle on 479) | 56.51% | reasoning preserved; class collapse on EQUAL/VAGUE; 30% chain-label decoupling at inference |
| **v4a** | multi-turn yes/no, no `<think>` | 12,471 | 54 MB | R1-Distill-14B | **Full FT**, 1 epoch | **0.7710** | **71.92%** | beats Yuan R1 (0.7017); EQUAL=0 emitted, VAGUE=2 emitted |
| **v4a-weight1** | same as v4a + `weight:1` on every assistant msg | 12,471 | 55 MB | R1-Distill-14B | Full FT, 1 epoch | **0.7726** | **72.04%** | Outcome 1 confirmed: +0.0016 F1 vs v4a — gate was already producing all-turn loss (or didn't fire on this model). v4a baseline stands. |
| **v4a-split** | each turn as its own conversation | 42,655 | 179 MB | R1-Distill-14B | Full FT, 1 epoch | _cancelled_ | — | switched to weight1 instead |
| v4b | multi-turn yes/no (teacher-correct subset only) | 7,519 | 32 MB | R1-Distill-14B | not trained | — | — | distillation subset; no gold-derived rows |
| v4c | multi-turn yes/no with `<think>...</Think>` rationale | 7,519 | 65 MB | R1-Distill-14B | not trained directly | — | — | original build (used as input to v4c-think) |
| **v4c-think** | same as v4c | 7,519 | 65 MB | R1-Distill-14B | Full FT, 1 epoch | **0.7522** | **67.62%** | **DOWN -1.88 F1 vs v4a.** Reasoning hurt: model over-commits VAGUE (88 emitted vs 2; recovered 12 VAGUE TPs but routed 54 BEFORE-gold and 16 AFTER-gold to VAGUE). BEFORE recall crashed 96% → 81%. wandb: tre-matres/tre-mt-think-v4c-fullft. job ft-935767a3-14b5. |
| v4c-native | multi-turn with native `reasoning` field | 7,519 | 65 MB | R1-Distill-14B | rejected at training start | — | — | Together rejects `reasoning` field for R1-Distill-Qwen-14B regardless of LoRA/full-FT. job ft-f72965f0-a6c3 (user_error, $0). |
| **v4a-Gemma3-12B** | same v4a data, different base | 12,471 | 54 MB | **google/gemma-3-12b-it** | Full FT, 1 epoch | **0.6176** | **57.59%** | **DOWN -15.34 F1 vs R1-Distill on same data.** Gemma collapsed AFTER recall to 23.4% (vs R1's 72.1%) and never routed Q1=Yes. R1-Distill's temporal reasoning prior is doing serious work that Gemma can't replicate. job ft-7c17b416-10a2. wandb: tre-matres/tre-v4a-gemma3-12b. (Note: 51/837 lost to endpoint replica failure mid-eval; defaulted to BEFORE in scoring; trend visible even in 786-pair subset.) |

## Baselines

| model | F1 |
|---|---|
| Yuan CoT DeepSeek-R1 | 0.7017 |
| LTM honest | ~0.80 |

## Data files (paths relative to GlobalZeroShotTRE/)

| version | training file | meta sidecar |
|---|---|---|
| v2 | `output/matres_train_sft_format_v2.jsonl` | — |
| v3 | `output/matres_train_sft_format_v3.jsonl` | — |
| v4a | `output/matres_train_v4a_full_directlabels.jsonl` | `.meta.jsonl` |
| v4a-weight1 | `output/matres_train_v4a_full_directlabels_weighted.jsonl` | (same as v4a) |
| v4a-split | `output/matres_train_v4a_split.jsonl` | (same as v4a) |
| v4b | `output/matres_train_v4b_subset_noThink.jsonl` | `.meta.jsonl` |
| v4c | `output/matres_train_v4c_subset_withThink.jsonl` | `.meta.jsonl` |

## Together IDs (this session)

| version | file_id | job_id | model output name | endpoint id |
|---|---|---|---|---|
| v4a | file-c005224a-3194-4785-8056-9b8a959aa1ba | ft-9beeaf96-0055 | `mikey641_af35/...-v4a-fullft-74dedad0` | endpoint-4da3b4d2-...-cfb70f9de5b4 (deleted) |
| v4a-split | file-4501b8c5-f1bb-472a-b6d7-53076e2ee02c | ft-ca81e5ff-d73e (cancelled) | `mikey641_af35/...-v4a-split-72abeca9` | n/a |
| v4a-weight1 | file-a23e527b-a72d-484e-8bb7-a4b8bd2429f4 | ft-f0252a4f-bb36 | `mikey641_af35/...-v4a-weight1-65ef1045` | endpoint-77170c06-...-7f73e370428b (deleted) |
| v4c-think | file-76e83067-816c-44f9-bcf3-dc3c21391fb3 | ft-935767a3-14b5 | `mikey641_af35/...-v4c-think-fullft-6ac9607a` | endpoint-cd2886f1-...-9931af60bb90 (deleted) |
| v4c-native | file-30383035-a511-4387-a18d-210fefc5d26b | ft-f72965f0-a6c3 (user_error, $0) | `mikey641_af35/...-v4c-reasoning-v4c-fullft-1b9d237e` | n/a |
| v4a-Gemma3-12B | file-c005224a-3194-4785-8056-9b8a959aa1ba | ft-7c17b416-10a2 | `mikey641_af35/gemma-3-12b-it-tre-v4a-gemma3-12b-84ef80a7` | endpoint-d75c9574-...-3ff6bece24d1 (deleted; previous endpoint-b93cdaa0 also deleted after replica died) |

## Hyperparameters (full-FT v4a family)

```
base = deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
training_method = sft
lora = False
n_epochs = 1
n_checkpoints = 1
learning_rate = 1e-5
lr_scheduler_type = cosine
min_lr_ratio = 0.1
scheduler_num_cycles = 1.0
warmup_ratio = 0.03
max_grad_norm = 1.0
weight_decay = 0.0
batch_size = 8 (provider cap)
suffix = tre-mt-yesno-v4a-{fullft|weight1|split}
```

## Eval scripts

| version | script |
|---|---|
| v3 | `scripts/eval/eval_v3_matres_full.py` |
| v4a (and family) | `scripts/eval/eval_v4a_yesno_chain.py` (configurable via `V4A_TRACES_PATH`, `V4A_DOT_DIR`, `V4A_DOT_FILE_NAME` env vars) |
| Test set | 837 MATRES test pairs via `load_test_pairs()` |

## Outputs (test eval)

| version | traces file | DOT file | F1 |
|---|---|---|---|
| v3 (partial 479) | `output/v3_epoch2_matres_test.traces.jsonl` | `output/v3_epoch2_matres_test_dot/matres_v3_epoch2_*.json` | 0.6058 emitted, 0.6378 oracle |
| v4a | `output/v4a_matres_test.traces.jsonl` | `output/v4a_matres_test_dot/matres_v4a.json` | 0.7710 |
| v4a-weight1 | `output/v4a_weight1_matres_test.traces.jsonl` | `output/v4a_weight1_matres_test_dot/matres_v4a_weight1.json` | 0.7726 |

## Open questions / next steps

- ~~v4a-weight1: does explicit `weight:1` change anything?~~ **Answered 2026-04-25**: F1 essentially flat (Δ +0.0016) but Q1=Yes routing collapsed 17 → 0. The change is real (different loss masking on early turns) but trade-balanced into ~same F1. v4a's 0.7710 baseline stands.
- ~~v4c-think: does reasoning supervision help?~~ **Answered 2026-04-25**: NO, on the teacher-correct subset alone it actively hurt (-1.88 F1, -4.30 acc). Model over-commits VAGUE due to reasoning-driven uncertainty. BUT the comparison is unfair: v4c had 60% the data and only "easy" pairs.
- ~~Together reasoning field on R1-Distill?~~ **Rejected for both LoRA and full FT** (2026-04-25 session). The `</Think>` workaround is the only path for reasoning supervision on this model.
- ~~Is R1-Distill the bottleneck (vs base model choice)?~~ **Answered 2026-04-25**: NO. Gemma-3-12B with same data → F1 0.6176 (-15.3 pts). R1-Distill's R1-derived temporal reasoning prior is doing significant work. Don't switch base.
- **v4d (next experiment idea):** full 12,471 multi-turn rows + reasoning EVERYWHERE (including for the 5,009 teacher-wrong pairs, where chains are *generated* to walk toward gold rather than the inverse-Yuan-tree shortcut used in v4a). Tests reasoning's value on a fair (full-data) baseline. Requires authoring 5,009 fresh chains via Gemini or full R1.
- Class rebalancing: EQUAL+VAGUE recovery is the obvious ceiling lift (currently 0% / 0.9% on v4a, 0% / 10.6% on v4c).
