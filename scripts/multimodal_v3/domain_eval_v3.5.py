# ==========================================================
# DOMAIN-SPECIFIC EVALUATION V5 — FAST (Batched + Light Model)
#
# Model     : Qwen/Qwen2.5-7B-Instruct  (~10x faster than 32B)
# Speed     : ~8-12 samples/min  →  ~2-3 hrs for 1554 samples
# Scoring   : Identical prompts/rubric to V4 (Table 7 verbatim)
# Changes   : Batched inference, lighter model, left-padding
#             for batch generation, everything else unchanged.
# ==========================================================

import json
import os
import re
import sys
import torch
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ══════════════════════════════════════════════════════════
#  SECTION 1 — CONFIG
# ══════════════════════════════════════════════════════════

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED_FILE = os.path.join(
    BASE_DIR,
    "baseline_qwen25vl",
    "outputs",
    "test_predictions.json"
)

SCORES_FILE = os.path.join(
    BASE_DIR,
    "baseline_qwen25vl",
    "outputs",
    "eval_scores_rubic_test_predictions.json"
)
# for my framework, I have to use the following paths instead of the above two
# PRED_FILE    = os.path.join(BASE_DIR, "outputs/multimodal_v3/test_predictions_v4.json")
# SCORES_FILE  = os.path.join(BASE_DIR, "outputs/multimodal_v3/eval_scores_v5_fast.json")
PARTIAL_FILE = SCORES_FILE.replace(".json", "_partial.json")

# ── Model: Qwen2.5-7B-Instruct ────────────────────────────
#   Same family as Qwen3-32B, same chat template, same
#   instruction-following quality for rubric scoring tasks.
#   ~65GB → ~15GB VRAM. Runs at ~8-12 samples/min batched.
#
#   Alternatives if you want even faster:
#     "Qwen/Qwen2.5-3B-Instruct"   # ~20 samples/min, lighter
#     "microsoft/Phi-3-mini-4k-instruct"  # very fast on CPU too
# ──────────────────────────────────────────────────────────
EVAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"

BATCH_SIZE        = 8     # increase to 16 if you have >24GB VRAM
SAVE_EVERY        = 50    # checkpoint every N scored samples
MAX_TOKENS        = 400   # enough for brief explanation + score lines
MAX_TOKENS_STRICT = 80    # strict retry: scores only
DTYPE             = torch.bfloat16

print("=" * 60)
print("  DOMAIN EVALUATION V5  —  Qwen2.5-7B  (Fast Batched)")
print(f"  Predictions : {PRED_FILE}")
print(f"  Output      : {SCORES_FILE}")
print(f"  Batch size  : {BATCH_SIZE}")
print("=" * 60 + "\n")

if not os.path.exists(PRED_FILE):
    print(f"ERROR: {PRED_FILE} not found.")
    sys.exit(1)

with open(PRED_FILE) as f:
    data = json.load(f)

total_samples = len(data)
print(f"[Data] Loaded {total_samples} predictions\n")

# ══════════════════════════════════════════════════════════
#  SECTION 2 — LOAD MODEL
# ══════════════════════════════════════════════════════════

print("[Model] Checking GPU memory...")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
else:
    print("  No CUDA GPUs detected — model will run on CPU (slow but works)")

print(f"\n[Model] Loading {EVAL_MODEL} tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    EVAL_MODEL,
    trust_remote_code=True,
)

# Left-padding is required for batched generation.
# The model was trained with right-padding but generation
# needs all sequences aligned on the RIGHT (new tokens side).
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[Model] Loading {EVAL_MODEL} weights (BF16)...")
model = AutoModelForCausalLM.from_pretrained(
    EVAL_MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

def get_model_device(m: torch.nn.Module) -> torch.device:
    try:
        return next(m.parameters()).device
    except StopIteration:
        return torch.device("cpu")

MODEL_DEVICE = get_model_device(model)
print(f"\n[Model] Ready.")
print(f"  First-layer device : {MODEL_DEVICE}")
print(f"  Dtype              : {DTYPE}\n")

# ══════════════════════════════════════════════════════════
#  SECTION 3 — PROMPTS  (Table 7 verbatim — unchanged)
# ══════════════════════════════════════════════════════════

CAPTION_SYSTEM = (
    "You are an advanced intelligent chatbot specialized in evaluating "
    "the accuracy of disaster scene captions that compare pre-disaster "
    "and post-disaster images."
)

CAPTION_USER_TMPL = """\
Your primary task is to meticulously compare the predicted caption with \
the ground truth caption and assess their factual consistency. \
To accomplish this, you will evaluate the captions across four key dimensions:

1. Damage Assessment Precision: Evaluate how accurately the elements \
mentioned in the predicted caption match the actual damage described in \
the ground truth caption. This measures whether the predicted details are \
correct (without considering comprehensiveness).

2. Damage Detail Recall: Assess how completely the predicted caption \
captures all the damage elements mentioned in the ground truth caption. \
This measures whether the prediction includes all relevant damage \
information from the ground truth.

3. Factual Correctness: Evaluate the absence of hallucinated content. \
Higher scores indicate fewer or no hallucinations, while lower scores \
indicate more hallucinations (facts, elements, or interpretations that \
do not exist in the ground truth caption or would not be visible in the \
images).

Please assign a score for each of these three dimensions, using an \
integer from 0 to 5, where 5 indicates perfect performance and 0 \
signifies poor performance. Accompany your assessments with brief \
explanations to clarify your scoring rationale.

GROUND TRUTH:
{gt}

PREDICTION:
{pred}

After your brief explanations, output the scores on separate lines \
exactly as shown:
DAP: <integer 0-5>
DDR: <integer 0-5>
FC: <integer 0-5>
"""

ADVICE_SYSTEM = (
    "You are an advanced intelligent evaluator specialized in assessing "
    "disaster recovery plans that compare recommended immediate and "
    "long-term recovery strategies following disasters."
)

ADVICE_USER_TMPL = """\
Your primary task is to meticulously compare the predicted recovery plan \
with the ground truth recovery plan and assess their factual consistency \
and strategic alignment. To accomplish this, you will evaluate the \
recovery plans across four key dimensions:

1. Recovery Necessity Recognition: Assess whether the predicted plan \
correctly recognizes if recovery actions are necessary. If the ground \
truth indicates no recovery is needed (e.g., "no discernible impact \
detected"), the prediction should similarly acknowledge this. Conversely, \
if the ground truth outlines necessary recovery actions, the prediction \
should not minimize or overlook the need for recovery.

2. Action Priority Precision: Evaluate how accurately the specific \
recovery actions mentioned in the predicted plan match the priorities \
described in the ground truth plans. This measures whether the predicted \
recovery actions are correct (without considering comprehensiveness). \
If no recovery is needed according to both plans, award full points.

3. Strategic Completeness: Assess how completely the predicted plan \
captures all the essential recovery strategies from the ground truth. \
If no recovery is needed according to both plans, award full points.

4. Implementation Feasibility: Evaluate the practicality and absence of \
unrealistic recommendations. Higher scores indicate realistic, \
implementable recovery actions, while lower scores indicate impractical \
suggestions or approaches that would be ineffective in the described \
disaster context. If no recovery is needed according to both plans, \
award full points.

Please assign a score for each of these four dimensions, using an \
integer from 0 to 5, where 5 indicates perfect performance and 0 \
signifies poor performance. Accompany your assessments with brief \
explanations to clarify your scoring rationale.

GROUND TRUTH:
{gt}

PREDICTION:
{pred}

After your brief explanations, output the scores on separate lines \
exactly as shown:
RN: <integer 0-5>
APP: <integer 0-5>
SC: <integer 0-5>
IF: <integer 0-5>
"""

# ══════════════════════════════════════════════════════════
#  SECTION 4 — TASK CLASSIFIER
# ══════════════════════════════════════════════════════════

ADVICE_KEYWORDS = {"restoration", "recovery"}

def is_advice_task(task: str) -> bool:
    t = task.lower()
    return any(kw in t for kw in ADVICE_KEYWORDS)

# ══════════════════════════════════════════════════════════
#  SECTION 5 — PROMPT BUILDER
# ══════════════════════════════════════════════════════════

def build_chat_text(gt: str, pred: str, task: str,
                    strict: bool = False) -> str:
    """Returns a fully formatted chat string ready for tokenization."""
    is_advice = is_advice_task(task)

    if strict:
        score_fmt = (
            "RN: <int>\nAPP: <int>\nSC: <int>\nIF: <int>"
            if is_advice else
            "DAP: <int>\nDDR: <int>\nFC: <int>"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a scoring assistant. "
                    "Output ONLY the score lines. No other text."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Score this. Output ONLY:\n{score_fmt}\n\n"
                    f"GROUND TRUTH:\n{gt}\n\nPREDICTION:\n{pred}"
                )
            }
        ]
    else:
        system = ADVICE_SYSTEM if is_advice else CAPTION_SYSTEM
        user   = (ADVICE_USER_TMPL if is_advice else CAPTION_USER_TMPL
                  ).format(gt=gt, pred=pred)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    # Qwen2.5 uses same apply_chat_template interface as Qwen3
    # enable_thinking not available in 2.5 — omit it
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

# ══════════════════════════════════════════════════════════
#  SECTION 6 — SCORE PARSER  (unchanged)
# ══════════════════════════════════════════════════════════

def extract_score(text: str, key: str) -> Optional[int]:
    pattern = rf"^{key}\s*:\s*([0-5])\b"
    match   = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_scores(text: str, task: str) -> Optional[dict]:
    keys   = ["RN", "APP", "SC", "IF"] if is_advice_task(task) \
             else ["DAP", "DDR", "FC"]
    scores = {}
    for k in keys:
        v = extract_score(text, k)
        if v is None:
            return None
        scores[k] = v
    return scores

# ══════════════════════════════════════════════════════════
#  SECTION 7 — BATCHED INFERENCE
#
#  Key changes vs V4:
#  - Process BATCH_SIZE samples at once via tokenizer padding
#  - Left-padding so new tokens always appear at the right
#  - Per-sample input_len tracked from attention_mask to
#    correctly slice only the generated tokens per sample
# ══════════════════════════════════════════════════════════

def run_batch_inference(batch: list[tuple[str, str, str]],
                        strict: bool = False) -> list[Optional[str]]:
    """
    batch : list of (gt, pred, task) tuples
    Returns list of decoded response strings (None on error per item).
    """
    texts = []
    for gt, pred, task in batch:
        texts.append(build_chat_text(gt, pred, task, strict=strict))

    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,   # cap prompt length to avoid OOM
        ).to(MODEL_DEVICE)

        input_lengths = inputs["attention_mask"].sum(dim=1)  # per-sample
        max_tok = MAX_TOKENS_STRICT if strict else MAX_TOKENS

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tok,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )

        results = []
        for i, out in enumerate(outputs):
            new_tokens = out[input_lengths[i]:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(decoded.strip())
        return results

    except Exception as e:
        tqdm.write(f"  [BatchError] {str(e)[:150]}")
        # Return None for every item in the batch so they go to retry
        return [None] * len(batch)

# ══════════════════════════════════════════════════════════
#  SECTION 8 — RESUME LOGIC
# ══════════════════════════════════════════════════════════

completed_indices: set = set()
all_scores: list       = []

if os.path.exists(PARTIAL_FILE):
    with open(PARTIAL_FILE) as f:
        partial = json.load(f)
    all_scores        = partial
    completed_indices = {r["index"] for r in partial}
    print(f"[Resume] {len(all_scores)} already scored. Skipping.\n")

# ══════════════════════════════════════════════════════════
#  SECTION 9 — EVALUATION LOOP  (batched)
# ══════════════════════════════════════════════════════════

pending = [
    (idx, item)
    for idx, item in enumerate(data)
    if idx not in completed_indices
]

print(f"[Eval] {len(pending)} samples to evaluate  |  batch={BATCH_SIZE}\n")

fail_empty     = 0
fail_parse     = 0
fail_model     = 0
retry_queue    = []
new_since_save = 0


def checkpoint():
    with open(PARTIAL_FILE, "w") as f:
        json.dump(all_scores, f, indent=4)


# ── PASS 1 : batched normal prompt ───────────────────────

def make_batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# Pre-filter empty items
valid_pending   = []
for idx, item in pending:
    gt   = item.get("ground_truth", "").strip()
    pred = item.get("prediction",   "").strip()
    if not gt or not pred:
        fail_empty += 1
    else:
        valid_pending.append((idx, item))

total_batches = (len(valid_pending) + BATCH_SIZE - 1) // BATCH_SIZE

with tqdm(total=len(valid_pending), desc="Pass 1 (batched)") as pbar:
    for batch_items in make_batches(valid_pending, BATCH_SIZE):

        # Build (gt, pred, task) tuples for inference
        meta    = []
        tuples  = []
        for idx, item in batch_items:
            gt   = item.get("ground_truth", "").strip()
            pred = item.get("prediction",   "").strip()
            task = item.get("task", "disaster caption")
            meta.append((idx, item, task))
            tuples.append((gt, pred, task))

        responses = run_batch_inference(tuples, strict=False)

        for (idx, item, task), response in zip(meta, responses):
            gt   = item.get("ground_truth", "").strip()
            pred = item.get("prediction",   "").strip()

            if response is None:
                fail_model += 1
                pbar.update(1)
                continue

            scores = parse_scores(response, task)

            if scores is None:
                tqdm.write(f"  [ParseRetry] idx={idx} | {repr(response[:80])}")
                retry_queue.append((idx, item))
                pbar.update(1)
                continue

            all_scores.append({
                "index":        idx,
                "task":         task,
                "is_sar":       item.get("is_sar", False),
                "scores":       scores,
                "ground_truth": gt,
                "prediction":   pred,
            })
            new_since_save += 1
            pbar.update(1)

        if new_since_save >= SAVE_EVERY:
            checkpoint()
            new_since_save = 0

checkpoint()

# ── PASS 2 : strict retry for parse failures ─────────────

if retry_queue:
    print(f"\n[Eval] Pass 2 — strict retry for {len(retry_queue)} parse failures\n")
    new_since_save = 0

    # Build tuples for retry
    retry_valid = []
    for idx, item in retry_queue:
        gt   = item.get("ground_truth", "").strip()
        pred = item.get("prediction",   "").strip()
        task = item.get("task", "disaster caption")
        retry_valid.append((idx, item, gt, pred, task))

    with tqdm(total=len(retry_valid), desc="Pass 2 (strict)") as pbar:
        for batch_chunk in make_batches(retry_valid, BATCH_SIZE):
            tuples = [(gt, pred, task) for _, _, gt, pred, task in batch_chunk]
            responses = run_batch_inference(tuples, strict=True)

            for (idx, item, gt, pred, task), response in zip(batch_chunk, responses):
                if response is None:
                    fail_model += 1
                    pbar.update(1)
                    continue

                scores = parse_scores(response, task)
                if scores is None:
                    tqdm.write(f"  [ParseFail] idx={idx}  giving up.")
                    fail_parse += 1
                    pbar.update(1)
                    continue

                all_scores.append({
                    "index":        idx,
                    "task":         task,
                    "is_sar":       item.get("is_sar", False),
                    "scores":       scores,
                    "ground_truth": gt,
                    "prediction":   pred,
                })
                new_since_save += 1
                pbar.update(1)

            if new_since_save >= SAVE_EVERY:
                checkpoint()
                new_since_save = 0

    checkpoint()

# ══════════════════════════════════════════════════════════
#  SECTION 10 — AGGREGATE + REPORT  (unchanged)
# ══════════════════════════════════════════════════════════

def avg(records: list, key: str) -> float:
    vals = [r["scores"][key] for r in records if key in r.get("scores", {})]
    return sum(vals) / len(vals) if vals else 0.0


caption_records = [r for r in all_scores if not is_advice_task(r["task"])]
advice_records  = [r for r in all_scores if     is_advice_task(r["task"])]

total_evaluated = len(all_scores)
total_skipped   = fail_empty + fail_model + fail_parse
coverage        = total_evaluated / total_samples * 100 if total_samples else 0.0

print("\n" + "=" * 60)
print("  DOMAIN METRICS V5  —  Table 7 / Table 8 Aligned")
print(f"  Evaluator : {EVAL_MODEL} (Transformers, BF16, Batched)")
print("=" * 60)
print(f"\n  Total samples  : {total_samples}")
print(f"  Evaluated      : {total_evaluated}  ({coverage:.1f}% coverage)")
print(f"  Skipped        : {total_skipped}")
print(f"    ├─ empty gt/pred  : {fail_empty}")
print(f"    ├─ model error    : {fail_model}")
print(f"    └─ parse fail     : {fail_parse}")

if caption_records:
    dap     = avg(caption_records, "DAP")
    ddr     = avg(caption_records, "DDR")
    fc      = avg(caption_records, "FC")
    cap_avg = (dap + ddr + fc) / 3

    print(f"\n  ── CAPTION TASK  (n={len(caption_records)}) ──────────")
    print(f"    DAP  Damage Assessment Precision : {dap:.3f} / 5")
    print(f"    DDR  Damage Detail Recall        : {ddr:.3f} / 5")
    print(f"    FC   Factual Correctness         : {fc:.3f} / 5")
    print(f"    ──────────────────────────────────────────")
    print(f"    AVG  (Table 8)                   : {cap_avg:.3f} / 5")

    sar_cap = [r for r in caption_records if     r["is_sar"]]
    opt_cap = [r for r in caption_records if not r["is_sar"]]

    if opt_cap:
        print(f"\n    Optical (n={len(opt_cap)}):  "
              f"DAP {avg(opt_cap,'DAP'):.3f}  "
              f"DDR {avg(opt_cap,'DDR'):.3f}  "
              f"FC  {avg(opt_cap,'FC'):.3f}")
    if sar_cap:
        print(f"    SAR     (n={len(sar_cap)}):  "
              f"DAP {avg(sar_cap,'DAP'):.3f}  "
              f"DDR {avg(sar_cap,'DDR'):.3f}  "
              f"FC  {avg(sar_cap,'FC'):.3f}")

if advice_records:
    rn      = avg(advice_records, "RN")
    app     = avg(advice_records, "APP")
    sc      = avg(advice_records, "SC")
    iff     = avg(advice_records, "IF")
    adv_avg = (rn + app + sc + iff) / 4

    print(f"\n  ── ADVICE TASK  (n={len(advice_records)}) ───────────")
    print(f"    RN   Recovery Necessity          : {rn:.3f} / 5")
    print(f"    APP  Action Priority Precision   : {app:.3f} / 5")
    print(f"    SC   Strategic Completeness      : {sc:.3f} / 5")
    print(f"    IF   Implementation Feasibility  : {iff:.3f} / 5")
    print(f"    ──────────────────────────────────────────")
    print(f"    AVG  (Table 8)                   : {adv_avg:.3f} / 5")
    print(f"\n    Table 8 cols → RN={rn:.3f}  APP={app:.3f}  SC={sc:.3f}")
    print(f"    (IF={iff:.3f} saved but not a Table 8 display column)")

print()

# ══════════════════════════════════════════════════════════
#  SECTION 11 — SAVE FINAL JSON  (unchanged structure)
# ══════════════════════════════════════════════════════════

final_output = {
    "evaluator":        EVAL_MODEL,
    "backend":          "transformers",
    "dtype":            "bfloat16",
    "batch_size":       BATCH_SIZE,
    "prompt_source":    "Table 7 verbatim",
    "predictions_file": PRED_FILE,
    "total_samples":    total_samples,
    "total_evaluated":  total_evaluated,
    "coverage_pct":     round(coverage, 2),
    "failures": {
        "empty_gt_or_pred": fail_empty,
        "model_error":      fail_model,
        "parse_fail":       fail_parse,
    },
    "caption": {
        "n":   len(caption_records),
        "DAP": round(avg(caption_records, "DAP"), 4) if caption_records else None,
        "DDR": round(avg(caption_records, "DDR"), 4) if caption_records else None,
        "FC":  round(avg(caption_records, "FC"),  4) if caption_records else None,
        "AVG": round(
                   (avg(caption_records, "DAP") +
                    avg(caption_records, "DDR") +
                    avg(caption_records, "FC")) / 3, 4
               ) if caption_records else None,
    },
    "advice": {
        "n":   len(advice_records),
        "RN":  round(avg(advice_records, "RN"),  4) if advice_records else None,
        "APP": round(avg(advice_records, "APP"), 4) if advice_records else None,
        "SC":  round(avg(advice_records, "SC"),  4) if advice_records else None,
        "IF":  round(avg(advice_records, "IF"),  4) if advice_records else None,
        "AVG": round(
                   (avg(advice_records, "RN")  +
                    avg(advice_records, "APP") +
                    avg(advice_records, "SC")  +
                    avg(advice_records, "IF")) / 4, 4
               ) if advice_records else None,
    },
    "per_sample": all_scores,
}

with open(SCORES_FILE, "w") as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

if os.path.exists(PARTIAL_FILE):
    os.remove(PARTIAL_FILE)

print(f"  Saved → {SCORES_FILE}")
print("  EVALUATION COMPLETE")