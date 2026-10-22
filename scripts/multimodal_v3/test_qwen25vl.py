# ==========================================================
# BASELINE: Qwen2.5-VL-7B-Instruct — Zero-Shot Inference
#
# Purpose
# -------
# This script evaluates the ORIGINAL pretrained Qwen2.5-VL-7B-Instruct
# model on the same test set used by the Rescue-VLM framework.
# It serves as the zero-shot baseline for AAAI paper comparison.
#
# What is IDENTICAL to evaluate_vlm_v3_5.py (framework script):
#   - test.json dataset
#   - image paths and loading
#   - SAR preprocessing pipeline (3-stage V4)
#   - sensor_tag() function and values
#   - map_task() function and values
#   - INSTRUCTION constant (exact string)
#   - prompt construction order (sensor_tag + task + INSTRUCTION)
#   - task-aware token budget (caption=512, recovery=260)
#   - lstrip_prefix output cleaning (not .replace)
#   - bleed-cut logic for recovery task
#   - section validation warnings
#   - partial save every 50 samples (resume support)
#
# What CHANGES vs the framework:
#   REMOVED:
#     - OpenCLIP ViT-B/16 encoder
#     - Temporal difference fusion [pre | post | diff]
#     - MultimodalProjectorV3
#     - LoRA adapter
#     - inputs_embeds injection
#   REPLACED WITH:
#     - Qwen2.5-VL-7B-Instruct native vision encoder
#     - Official AutoProcessor + apply_chat_template()
#     - process_vision_info() for image tokenisation
#     - Standard model.generate() with pixel_values + input_ids
#
#   OUTPUT FORMAT CHANGE (per request):
#     The FINAL predictions file (test_predictions.json) now contains
#     ONLY ground_truth + prediction per sample:
#       [
#         { "ground_truth": "...", "prediction": "..." },
#         ...
#       ]
#     All bookkeeping fields (task, is_sar, sections_ok, _idx) are still
#     tracked internally and kept in the PARTIAL file so resume support
#     and section-validation warnings keep working exactly as before —
#     they are just stripped out at the very last step, when the final
#     file is written.
#
# Folder layout expected:
#   Rescue_VLM_Ultimate_Version3.5/
#     data/
#       test.json
#       train_images/   (or whatever image_root the paths point into)
#     models/
#       Qwen2.5-VL-7B-Instruct/   ← download here beforehand
#     baseline_qwen25vl/
#       outputs/                   ← created automatically
#         test_predictions.json   ← this script writes here
#
# Run:
#   cd Rescue_VLM_Ultimate_Version3.5/baseline_qwen25vl
#   python test_qwen25vl.py
#
# Resume:
#   If the run is interrupted, re-running the script resumes from
#   the last completed sample (partial save every SAVE_EVERY samples).
# ==========================================================

import sys
import os
import gc
import json
import time
import torch
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

# ── transformers ──────────────────────────────────────────
# Try the recommended class name first; fall back to the
# generic alias that newer transformers versions expose.
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVL
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as QwenVL
    except ImportError:
        from transformers import AutoModelForVision2Seq as QwenVL

from transformers import AutoProcessor

# qwen_vl_utils ships alongside Qwen2.5-VL in the HF repo.
# If not installed, install with:  pip install qwen-vl-utils
from qwen_vl_utils import process_vision_info


# ══════════════════════════════════════════════════════════
#  SECTION 1 — PATHS
#
#  This script lives at:
#    baseline_qwen25vl/test_qwen25vl.py
#
#  BASE_DIR resolves to the project root:
#    Rescue_VLM_Ultimate_Version3.5/
# ══════════════════════════════════════════════════════════

# baseline_qwen25vl/ → parent = Rescue_VLM_Ultimate_Version3.5/
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = os.path.join(BASE_DIR, "data")
TEST_JSON  = os.path.join(DATA_DIR, "test.json")

MODEL_DIR  = os.path.join(BASE_DIR, "models", "Qwen2.5-VL-7B-Instruct")

OUTPUT_DIR = os.path.join(BASE_DIR, "baseline_qwen25vl", "outputs")
PRED_FILE  = os.path.join(OUTPUT_DIR, "test_predictions.json")
PARTIAL_FILE = PRED_FILE.replace(".json", "_partial.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
#  SECTION 2 — CONFIG
# ══════════════════════════════════════════════════════════

device = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16

# Task-aware token budgets — IDENTICAL to framework.
# Caption:  512 — six sections fill naturally; CONCLUSION stops generation.
# Recovery: 260 — two sections only; more tokens causes generation bleed
#                  (model loops and repeats IMMEDIATE_RECOVERY again).
MAX_NEW_TOKENS = {
    "caption":  512,
    "recovery": 260,
}

SAVE_EVERY = 50   # write partial file every N completed samples

print("=" * 60)
print("  BASELINE: Qwen2.5-VL-7B-Instruct  (Zero-Shot)")
print("=" * 60)
print(f"  Device      : {device}")
print(f"  Dtype       : {DTYPE}")
print(f"  Model dir   : {MODEL_DIR}")
print(f"  Test JSON   : {TEST_JSON}")
print(f"  Output file : {PRED_FILE}")
print("=" * 60 + "\n")

# Sanity check
if not os.path.isdir(MODEL_DIR):
    print(f"[ERROR] Model not found at: {MODEL_DIR}")
    print("  Download with:")
    print(f"    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct "
          f"--local-dir {MODEL_DIR}")
    sys.exit(1)

if not os.path.exists(TEST_JSON):
    print(f"[ERROR] Test JSON not found at: {TEST_JSON}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════
#  SECTION 3 — CONSTANTS
#
#  INSTRUCTION is copied verbatim from train_vlm_v3_dgx.py.
#  DO NOT change it — the prompt in this baseline must be
#  byte-identical to the framework prompt for a fair comparison.
# ══════════════════════════════════════════════════════════

INSTRUCTION = (
    "Carefully analyze differences between pre- and post-disaster images. "
    "Detect even minor structural or environmental changes. "
    "Generate structured report:\n"
)


# ══════════════════════════════════════════════════════════
#  SECTION 4 — SENSOR TAG
#
#  Identical to sensor_tag() in train_vlm_v3_dgx.py and
#  evaluate_vlm_v3_5.py.  Unicode escape kept for safety.
# ══════════════════════════════════════════════════════════

def sensor_tag(is_sar: bool) -> str:
    if is_sar:
        return "[SENSOR: OPTICAL\u2192SAR]\n"
    return "[SENSOR: OPTICAL\u2192OPTICAL]\n"


# ══════════════════════════════════════════════════════════
#  SECTION 5 — TASK MAP
#
#  Identical to map_task() in evaluate_vlm_v3_5.py.
#  Defaults to caption on unknown task strings (safe).
# ══════════════════════════════════════════════════════════

def map_task(t: str) -> str:
    t = t.lower().strip()
    if "restoration" in t or "recovery" in t:
        return "### TASK: DISASTER RESTORATION ADVICE ###\n"
    return "### TASK: DISASTER CAPTION ###\n"


# ══════════════════════════════════════════════════════════
#  SECTION 6 — SAR PREPROCESSING (V4 — 3 stages)
#
#  Identical to preprocess_sar() in dataset_loader V4 and
#  evaluate_vlm_v3_5.py. The same pixels that the framework
#  saw during training are fed to the baseline here, ensuring
#  the only variable between the two systems is the model.
#
#  Stage 1: MedianFilter(3) — despeckle before intensity math
#  Stage 2: log1p            — compress 40dB range → Gaussian
#  Stage 3: percentile [1,99] stretch on log values
# ══════════════════════════════════════════════════════════

def preprocess_sar(pil_img: Image.Image) -> Image.Image:
    img_gray = ImageOps.grayscale(pil_img)
    img_desp = img_gray.filter(ImageFilter.MedianFilter(size=3))   # Stage 1
    arr      = np.array(img_desp, dtype=np.float32)
    arr      = np.log1p(arr)                                        # Stage 2
    lo, hi   = np.percentile(arr, 1), np.percentile(arr, 99)       # Stage 3
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    arr     = np.clip(arr * 255, 0, 255).astype(np.uint8)
    arr_rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr_rgb, mode="RGB")


# ══════════════════════════════════════════════════════════
#  SECTION 7 — OUTPUT CLEANING HELPERS
#
#  lstrip_prefix: strips a prefix from the start of a string
#  exactly once. Used instead of .replace() to avoid removing
#  the fragment if it genuinely appears inside the prediction.
#
#  clean_prediction: removes prompt echo from the start of
#  the decoded text and applies bleed-cut for recovery task.
# ══════════════════════════════════════════════════════════

def lstrip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s


def clean_prediction(raw: str, task_str: str, tag: str, task: str) -> str:
    """
    Strip prompt echo from the generated text and apply the
    recovery bleed-cut.  Logic is identical to evaluate_vlm_v3_5.py.
    """
    pred = raw.strip()

    # Strip prompt fragments from the start (in training order)
    for fragment in [
        "[SENSOR: OPTICAL\u2192SAR]\n",
        "[SENSOR: OPTICAL\u2192OPTICAL]\n",
        tag,
        "### TASK: DISASTER RESTORATION ADVICE ###\n",
        "### TASK: DISASTER CAPTION ###\n",
        task_str,
        INSTRUCTION,
        "Generate structured report:\n",
    ]:
        pred = lstrip_prefix(pred, fragment)

    pred = pred.strip()

    # Recovery bleed-cut: cut anything after LONG_TERM_RECOVERY section ends
    if "recovery" in task.lower() or "restoration" in task.lower():
        if "LONG_TERM_RECOVERY:" in pred:
            before, after = pred.split("LONG_TERM_RECOVERY:", 1)
            bleed_markers = [
                "(END)",
                "Carefully analyze",
                "IMMEDIATE_RECOVERY:",
                "### TASK:",
                "[SENSOR:",
                tag,
            ]
            cut_pos = len(after)
            for marker in bleed_markers:
                pos = after.find(marker)
                if pos != -1 and pos < cut_pos:
                    cut_pos = pos
            after = after[:cut_pos].strip()
            pred  = before.strip() + "\nLONG_TERM_RECOVERY: " + after

    return pred


# ══════════════════════════════════════════════════════════
#  SECTION 8 — LOAD DATASET
# ══════════════════════════════════════════════════════════

with open(TEST_JSON, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"[Data] Test samples loaded: {len(test_data)}\n")


# ══════════════════════════════════════════════════════════
#  SECTION 9 — RESUME LOGIC
#
#  Completed sample indices are read from the partial file
#  so a re-run skips already-processed samples. The partial
#  file keeps ALL bookkeeping fields (_idx, task, is_sar,
#  sections_ok) — only the FINAL file is trimmed down to
#  {ground_truth, prediction}.
# ══════════════════════════════════════════════════════════

results            = []
completed_indices  = set()

if os.path.exists(PARTIAL_FILE):
    with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
        partial = json.load(f)
    results           = partial
    completed_indices = {r["_idx"] for r in partial}
    print(f"[Resume] {len(results)} samples already done — skipping.\n")


# ══════════════════════════════════════════════════════════
#  SECTION 10 — LOAD QWEN2.5-VL
#
#  The model is loaded in bfloat16 with device_map="auto"
#  so PyTorch handles GPU placement automatically.
#  Flash Attention 2 is enabled when the installed version
#  of transformers supports it; otherwise falls back silently.
# ══════════════════════════════════════════════════════════

print("[Model] Loading Qwen2.5-VL-7B-Instruct...")

processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
)

# Try to enable flash_attention_2 (faster, lower memory).
# If not available, fall back to eager attention.
try:
    model = QwenVL.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    print("  Flash Attention 2 enabled.")
except Exception:
    model = QwenVL.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    print("  Flash Attention 2 not available — using eager attention.")

model.eval()
print("[Model] Ready.\n")


# ══════════════════════════════════════════════════════════
#  SECTION 11 — INFERENCE LOOP
#
#  Per-sample pipeline:
#    1.  Resolve image paths (handle Windows backslashes)
#    2.  Load pre-disaster image as RGB (always optical)
#    3.  Load post-disaster image; apply SAR pipeline if SAR
#    4.  Build prompt: sensor_tag + map_task + INSTRUCTION
#    5.  Build Qwen messages with interleaved image+text
#        (matches DisasterM3 paper repo run_vllm.py format)
#    6.  apply_chat_template() → text string
#    7.  process_vision_info() → image tensors
#    8.  processor() → input_ids, pixel_values, attention_mask
#    9.  model.generate()
#   10.  Decode → strip input tokens → clean prediction
#   11.  Validate section headers
#   12.  Append to results (full bookkeeping); partial save
#        every SAVE_EVERY
#
#  NOTE on Qwen image format:
#    Qwen2.5-VL expects images supplied as PIL Images (not file
#    paths) when running with transformers (not vLLM).  The
#    official process_vision_info() call handles resizing and
#    pixel-value extraction internally.
# ══════════════════════════════════════════════════════════

skipped   = 0
sar_count = 0
sections_ok_count = 0
new_since_save = 0


def do_partial_save():
    with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


loop = tqdm(enumerate(test_data), total=len(test_data), desc="Baseline inference")

for idx, item in loop:

    # Skip already completed samples (resume support)
    if idx in completed_indices:
        continue

    try:
        # ── 1. Resolve paths ──────────────────────────────
        pre_rel  = item["pre_image_path"].replace("\\", "/")
        post_rel = item["post_image_path"].replace("\\", "/")
        pre_path  = os.path.join(DATA_DIR, pre_rel)
        post_path = os.path.join(DATA_DIR, post_rel)

        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            tqdm.write(f"  [SKIP] Missing images at idx={idx}")
            skipped += 1
            continue

        # ── 2. Load pre image (always optical RGB) ────────
        with Image.open(pre_path) as img:
            pre_img = img.convert("RGB")

        # ── 3. Load post image; SAR preprocessing ─────────
        post_type = item.get("post_image_type", "Optical").strip().upper()
        is_sar    = (post_type == "SAR")
        if is_sar:
            with Image.open(post_path) as img:
                post_img = preprocess_sar(img)
            sar_count += 1
        else:
            with Image.open(post_path) as img:
                post_img = img.convert("RGB")

        # ── 4. Build text prompt ───────────────────────────
        task      = item.get("task", "disaster caption")
        tag       = sensor_tag(is_sar)
        task_str  = map_task(task)
        prompt    = tag + task_str + INSTRUCTION

        # ── 5. Build Qwen messages ─────────────────────────
        # Format follows DisasterM3 paper repo run_vllm.py:
        # [text_before_image1] + [image1] + [text_between] + [image2] + [text_after]
        # For caption/recovery the prompt surrounds the two images.
        # We split at natural boundaries: before pre-image, between images, after post-image.
        messages = [
            {
                "role": "user",
                "content": [
                    # Sensor + task header + start of instruction before images
                    {"type": "text",  "text": prompt},
                    # Pre-disaster image
                    {"type": "image", "image": pre_img},
                    # Small label so the model knows this is the pre-disaster image
                    {"type": "text",  "text": "\nPost-disaster image:"},
                    # Post-disaster image
                    {"type": "image", "image": post_img},
                ],
            }
        ]

        # ── 6. Apply chat template ─────────────────────────
        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ── 7. Process vision info ─────────────────────────
        image_inputs, video_inputs = process_vision_info(messages)

        # ── 8. Tokenise ────────────────────────────────────
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        input_len = inputs["input_ids"].shape[1]

        # ── 9. Token budget ────────────────────────────────
        task_key   = (
            "recovery"
            if ("recovery" in task.lower() or "restoration" in task.lower())
            else "caption"
        )
        max_tokens = MAX_NEW_TOKENS[task_key]

        # ── 10. Generate ───────────────────────────────────
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens     = max_tokens,
                do_sample          = False,
                repetition_penalty = 1.1,
                eos_token_id       = processor.tokenizer.eos_token_id,
            )

        # Strip the input tokens — keep only the newly generated tokens
        new_ids = generated_ids[0][input_len:]
        raw     = processor.tokenizer.decode(
            new_ids, skip_special_tokens=True
        ).strip()

        # ── 11. Clean output ───────────────────────────────
        pred = clean_prediction(raw, task_str, tag, task)

        # ── 12. Validate structure ─────────────────────────
        if "recovery" in task.lower() or "restoration" in task.lower():
            expected_keys = ["IMMEDIATE_RECOVERY:", "LONG_TERM_RECOVERY:"]
        else:
            expected_keys = ["BUILDING:", "ROAD:", "CONCLUSION:"]

        missing_keys = [k for k in expected_keys if k not in pred]

        if not pred:
            tqdm.write(f"  [WARN] Empty prediction idx={idx}")
        elif missing_keys:
            tqdm.write(
                f"  [WARN] idx={idx} missing sections {missing_keys} "
                f"(len={len(pred)} chars)"
            )

        if len(missing_keys) == 0:
            sections_ok_count += 1

        # ── 13. Ground truth ───────────────────────────────
        ground_truth = (
            item.get("ground_truth")
            or item.get("training_answer")
            or ""
        )

        # ── 14. Store result ───────────────────────────────
        # _idx, task, is_sar, sections_ok are kept here for resume
        # support and diagnostics; they are stripped out at final
        # save time so the delivered file only has
        # {ground_truth, prediction}.
        results.append({
            "_idx":         idx,
            "task":         task.strip(),
            "is_sar":       is_sar,
            "ground_truth": ground_truth.strip(),
            "prediction":   pred,
            "sections_ok":  len(missing_keys) == 0,
        })
        new_since_save += 1

        # ── 15. Periodic partial save ──────────────────────
        if new_since_save >= SAVE_EVERY:
            do_partial_save()
            new_since_save = 0

        # ── 16. GPU memory housekeeping ────────────────────
        del inputs, generated_ids, new_ids
        if idx % 100 == 0 and idx > 0:
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        tqdm.write(f"  [ERROR] idx={idx}: {e}")
        skipped += 1


# ══════════════════════════════════════════════════════════
#  SECTION 12 — FINAL SAVE
#
#  The delivered predictions file is trimmed down to EXACTLY:
#    [
#      { "ground_truth": "...", "prediction": "..." },
#      ...
#    ]
#  All bookkeeping fields (_idx, task, is_sar, sections_ok)
#  are dropped here — they only ever lived in `results` /
#  the partial file to support resume + the warnings above.
# ══════════════════════════════════════════════════════════

final_results = [
    {
        "ground_truth": r["ground_truth"],
        "prediction":   r["prediction"],
    }
    for r in results
]

with open(PRED_FILE, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4, ensure_ascii=False)

# Clean up partial file if it exists
if os.path.exists(PARTIAL_FILE):
    os.remove(PARTIAL_FILE)


# ══════════════════════════════════════════════════════════
#  SECTION 13 — SUMMARY
#
#  Uses the running counters tracked during the loop
#  (sar_count, sections_ok_count) rather than reading them
#  back off final_results, since final_results no longer
#  carries task/is_sar/sections_ok.
# ══════════════════════════════════════════════════════════

optical_count = len(final_results) - sar_count
sections_warn = len(final_results) - sections_ok_count

print("\n" + "=" * 60)
print("  BASELINE PREDICTIONS SAVED")
print("=" * 60)
print(f"  File         : {PRED_FILE}")
print(f"  Evaluated    : {len(final_results)}")
print(f"    Optical    : {optical_count}")
print(f"    SAR        : {sar_count}")
print(f"  Skipped      : {skipped}")
print(f"  Sections OK  : {sections_ok_count} / {len(final_results)}")
if sections_warn:
    print(f"  [WARN] {sections_warn} predictions missing expected section headers.")
print()
print("  Next step — run domain evaluation:")
print(f"    cd {os.path.dirname(BASE_DIR)}")
print(f"    python Rescue_VLM_Ultimate_Version3.5/scripts/"
      f"multimodal_v3/domain_eval_v3_5.py \\")
print(f"      --pred_file {PRED_FILE}")
print()
print("  Or point domain_eval_v3_5.py PRED_FILE to the path above.")
print("=" * 60)
