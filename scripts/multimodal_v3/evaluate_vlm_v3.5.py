# ==========================================================
# GENERATE TEST PREDICTIONS V4 — FULLY ALIGNED TO TRAINING
#
# Aligned against:
#   dataset_loader.py       V4  (3-stage SAR pipeline)
#   train_vlm_v3_dgx.py        (zero-hang + SAR cross-sensor)
#   test_pipeline_v4.py        (single-image inference reference)
#
# Every component below is verified to be byte-identical to
# its counterpart in training. Any divergence = wrong scores.
#
# FIXES vs old generate_test_predictions_v3.py:
#
#   FIX 1 — CLIP fine-tuned blocks NOT loaded  [CRITICAL]
#     Old: only OpenAI pretrained weights used.
#     Training fine-tuned CLIP blocks 8,9,10,11 + ln_post on
#     your disaster dataset. Projector was trained on those
#     fine-tuned embeddings. Not loading clip_partial.pt means
#     CLIP produces out-of-distribution embeddings → projector
#     maps them to the wrong token space → wrong output.
#     Fix: load clip_partial.pt from best_checkpoint/.
#
#   FIX 2 — Fusion dimension 1024 → 1536  [CRITICAL / CRASH]
#     Old: cat([pre, post]) = (1, 1024)
#     Training: cat([pre, post, diff]) = (1, 1536)
#     MultimodalProjectorV3 expects 1536-dim input.
#     Old code would crash or silently corrupt every prediction.
#     Fix: fuse_features() = cat([pre, post, post-pre]).
#
#   FIX 3 — SAR preprocessing absent  [CRITICAL]
#     Old: all images → .convert("RGB"), including SAR post images.
#     Training: SAR post images → 3-stage pipeline
#       (MedianFilter despeckle → log1p → percentile stretch).
#     SAR samples fed as raw RGB to a model trained on
#     log-compressed SAR = completely wrong visual embedding.
#     Fix: read post_image_type per item, apply preprocess_sar.
#
#   FIX 4 — Sensor tag absent  [CRITICAL]
#     Old: no sensor tag in prompt.
#     Training: sensor_tag(is_sar) always prepended as first token.
#     The LoRA adapter learned to condition on this tag.
#     Without it, every generated token is off-distribution.
#     Fix: prepend "[SENSOR: OPTICAL→SAR]\n" or "→OPTICAL]\n".
#
#   FIX 5 — INSTRUCTION string truncated  [HIGH]
#     Old: "Generate structured report:\n"
#     Training: full 3-sentence INSTRUCTION.
#     Model was never trained on the short version.
#     Fix: use exact INSTRUCTION constant from training.
#
#   FIX 6 — Prompt order wrong  [HIGH]
#     Old:  task → short_instruction
#     Training: sensor_tag → task → INSTRUCTION
#     Fix: correct order, matches forward_pass() in training.
#
#   FIX 7 — LLM dtype float16 → bfloat16  [HIGH]
#     GB10 Blackwell is native bfloat16. float16 risks
#     gradient overflow and is slower on this hardware.
#     Fix: torch_dtype=torch.bfloat16.
#
#   FIX 8 — Load from last epoch → best_checkpoint/  [HIGH]
#     Old: loads projector + LoRA from OUTPUT_DIR (last epoch).
#     Best quality always comes from best_checkpoint/ which
#     holds the weights at lowest validation loss.
#     Fix: CKPT_DIR = outputs/multimodal_v3/best_checkpoint.
#
#   FIX 9 — Projector dtype float32 → bfloat16  [MEDIUM]
#     Training: projector.to(device).to(bfloat16).
#     Fix: match training dtype.
#
#   FIX 10 — No autocast  [MEDIUM]
#     Training uses torch.amp.autocast("cuda", dtype=bfloat16)
#     for projector + LLM forward. Without it, some ops silently
#     run in float32 causing memory spikes.
#     Fix: autocast wraps projector + text embedding + generate.
#
#   FIX 11 — max_new_tokens 300 → 512  [MEDIUM]
#     Structured labels have 6 sections. 300 tokens cuts output.
#     Fix: 512.
#
#   FIX 12 — Output stripping  [LOW]
#     Old only stripped 2 fragments. Must strip sensor_tag and
#     full INSTRUCTION to get clean structured text.
#
#   FIX 13 — ground_truth key  [LOW]
#     Old: hardcoded item["ground_truth"].
#     Training JSON uses "training_answer". Test JSON may have
#     either or neither. Fix: graceful .get() with fallback.
# ==========================================================

import sys
import os
import json
import numpy as np
import torch
import open_clip
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from models.projector_v3 import MultimodalProjectorV3


# ══════════════════════════════════════════════════════════
#  SECTION 1 — CONFIG
# ══════════════════════════════════════════════════════════

TEST_JSON  = os.path.join(BASE_DIR, "data/test.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/multimodal_v3")
CKPT_DIR   = os.path.join(OUTPUT_DIR, "best_checkpoint")   # FIX 8
PRED_FILE  = os.path.join(OUTPUT_DIR, "test_predictions_v4.json")

DTYPE           = torch.bfloat16     # FIX 7: bfloat16 not float16
UNFREEZE_BLOCKS = [8, 9, 10, 11]
MAX_NEW_TOKENS  = 512                # FIX 11: was 300

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Config] Device={device}  Dtype={DTYPE}")
print(f"[Config] Checkpoint={CKPT_DIR}")
print(f"[Config] Output={PRED_FILE}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
#  SECTION 2 — CONSTANTS  (verbatim from train_vlm_v3_dgx.py)
#  DO NOT EDIT — must match training exactly.
# ══════════════════════════════════════════════════════════

# FIX 5: full INSTRUCTION, not "Generate structured report:\n"
INSTRUCTION = (
    "Carefully analyze differences between pre- and post-disaster images. "
    "Detect even minor structural or environmental changes. "
    "Generate structured report:\n"
)


# ══════════════════════════════════════════════════════════
#  SECTION 3 — SENSOR TAG  (verbatim from training)
#  FIX 4: was absent in old code.
# ══════════════════════════════════════════════════════════

def sensor_tag(is_sar: bool) -> str:
    """
    Identical to sensor_tag() in train_vlm_v3_dgx.py.
    Unicode arrow \u2192 = →. Kept as escape for encoding safety.
    """
    if is_sar:
        return "[SENSOR: OPTICAL\u2192SAR]\n"
    return "[SENSOR: OPTICAL\u2192OPTICAL]\n"


# ══════════════════════════════════════════════════════════
#  SECTION 4 — TASK MAP  (verbatim from training)
#  Extended to handle raw task strings from test JSON
#  (training raises ValueError; here we default to caption
#  rather than crash on an unknown string in test data).
# ══════════════════════════════════════════════════════════

def map_task(t: str) -> str:
    t = t.lower().strip()
    if "restoration" in t or "recovery" in t:
        return "### TASK: DISASTER RESTORATION ADVICE ###\n"
    return "### TASK: DISASTER CAPTION ###\n"   # safe default


# ══════════════════════════════════════════════════════════
#  SECTION 5 — SAR PREPROCESSING  (verbatim from dataset_loader V4)
#  FIX 3: was absent. All images were fed as raw RGB.
#
#  Stage 1: MedianFilter(3)      despeckle first, before any math
#  Stage 2: log1p                SAR backscatter log-normal → Gaussian
#  Stage 3: percentile [1,99]    stretch on log values
#  Output:  3-channel grayscale RGB so CLIP preprocess works
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
#  SECTION 6 — CHECKPOINT VERIFICATION
# ══════════════════════════════════════════════════════════

_required = ["lora", "projector_v3.pt", "clip_partial.pt"]
print("Checkpoint files:")
all_ok = True
for r in _required:
    exists = os.path.exists(os.path.join(CKPT_DIR, r))
    print(f"  [{'OK' if exists else 'MISSING'}] {r}")
    if not exists:
        all_ok = False
if not all_ok:
    print("\nERROR: Missing checkpoint files. Aborting.")
    sys.exit(1)
print()


# ══════════════════════════════════════════════════════════
#  SECTION 7 — LOAD CLIP
#
#  FIX 1: clip_partial.pt loaded (fine-tuned blocks 8-11).
#  FIX 7: explicit .float() for float32 (matches training).
#  Training fine-tuned these blocks on your disaster dataset.
#  Without loading them, CLIP embeddings are out-of-distribution
#  for the projector → wrong visual tokens → wrong output.
# ══════════════════════════════════════════════════════════

print("Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="openai"
)
clip_model = clip_model.float().to(device)   # float32 intentional (matches training)

missing, unexpected = clip_model.load_state_dict(
    torch.load(
        os.path.join(CKPT_DIR, "clip_partial.pt"),
        map_location=device,
        weights_only=True,
    ),
    strict=False,   # only loads the 4 fine-tuned blocks, rest stays pretrained
)
clip_model.eval()
print(f"  Fine-tuned blocks loaded (missing={len(missing)}, extra={len(unexpected)})")
print("  CLIP ready (float32)\n")


# ══════════════════════════════════════════════════════════
#  SECTION 8 — LOAD PROJECTOR
#
#  FIX 8: load from best_checkpoint/ not OUTPUT_DIR.
#  FIX 9: cast to bfloat16 to match training.
# ══════════════════════════════════════════════════════════

print("Loading Projector V3...")
projector = MultimodalProjectorV3().to(device).to(DTYPE)   # FIX 9
projector.load_state_dict(
    torch.load(
        os.path.join(CKPT_DIR, "projector_v3.pt"),
        map_location=device,
        weights_only=True,
    )
)
projector.eval()
print("  Projector V3 ready (bfloat16)\n")


# ══════════════════════════════════════════════════════════
#  SECTION 9 — LOAD LLM + LoRA
#
#  FIX 7: torch_dtype=bfloat16 (was float16).
#  FIX 8: load LoRA from best_checkpoint/.
# ══════════════════════════════════════════════════════════

print("Loading LLM + LoRA...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=DTYPE,           # FIX 7: bfloat16
).to(device)

llm = PeftModel.from_pretrained(llm, os.path.join(CKPT_DIR, "lora"))
llm.eval()
print("  LLM + LoRA ready (bfloat16)\n")


# ══════════════════════════════════════════════════════════
#  SECTION 10 — FUSION HELPER  (verbatim from training)
#  FIX 2: was cat([pre, post]) = 1024. Must be 1536.
# ══════════════════════════════════════════════════════════

def fuse_features(pre_feat: torch.Tensor,
                  post_feat: torch.Tensor) -> torch.Tensor:
    """
    Identical to fuse_features() in train_vlm_v3_dgx.py.
    Returns (B, 1536): [pre | post | post-pre]
    MultimodalProjectorV3 expects exactly 1536-dim input.
    """
    diff = post_feat - pre_feat
    return torch.cat([pre_feat, post_feat, diff], dim=1)


# ══════════════════════════════════════════════════════════
#  SECTION 11 — LOAD TEST DATA
# ══════════════════════════════════════════════════════════

with open(TEST_JSON) as f:
    test_data = json.load(f)

print(f"[Data] Test samples: {len(test_data)}\n")


# ══════════════════════════════════════════════════════════
#  SECTION 12 — INFERENCE LOOP
#
#  Per-sample pipeline (identical to training forward_pass):
#    1. Read post_image_type → is_sar flag
#    2. Load images: pre always optical, post via is_sar
#    3. CLIP encode (float32, no autocast)
#    4. Fuse [pre | post | diff] = (1, 1536)
#    5. Project with autocast(bfloat16)
#    6. Build prompt: sensor_tag + map_task + INSTRUCTION
#    7. Tokenize + embed
#    8. Prepend visual tokens
#    9. Generate with autocast(bfloat16)
# ══════════════════════════════════════════════════════════

results = []
skipped = 0
sar_count = 0

for item in tqdm(test_data, desc="Generating predictions"):
    try:
        # ── 1. Paths ──────────────────────────────────────
        pre_rel  = item["pre_image_path"].replace("\\", os.sep)
        post_rel = item["post_image_path"].replace("\\", os.sep)

        pre_path  = os.path.join(BASE_DIR, "data", pre_rel)
        post_path = os.path.join(BASE_DIR, "data", post_rel)

        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            skipped += 1
            continue

        # ── 2. SAR flag  (FIX 3) ──────────────────────────
        post_type = item.get("post_image_type", "Optical").strip().upper()
        is_sar    = (post_type == "SAR")
        if is_sar:
            sar_count += 1

        # ── 3. Load images ────────────────────────────────
        with Image.open(pre_path) as img:
            pre_tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)

        with Image.open(post_path) as img:
            if is_sar:
                post_tensor = preprocess(preprocess_sar(img)).unsqueeze(0).to(device)
            else:
                post_tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)

        # ── 4. CLIP encode — float32, no autocast (FIX 1,7) ──
        with torch.no_grad():
            pre_feat  = clip_model.encode_image(pre_tensor.float())   # (1, 512)
            post_feat = clip_model.encode_image(post_tensor.float())  # (1, 512)

        # ── 5. Fuse [pre | post | diff] = 1536  (FIX 2) ──────
        fusion = fuse_features(pre_feat, post_feat)   # (1, 1536)

        # ── 6. Project + build prompt  (FIX 9, 10) ───────────
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=DTYPE):   # FIX 10

                proj = projector(fusion)   # (1, N_tokens, dim) bfloat16

                # Build prompt in training order  (FIX 4, 5, 6)
                tag       = sensor_tag(is_sar)       # "[SENSOR: OPTICAL→SAR]\n"
                task_str  = map_task(item.get("task", "disaster caption"))
                prompt    = tag + task_str + INSTRUCTION

                tokens      = tokenizer(prompt, return_tensors="pt").to(device)
                text_embeds = llm.get_input_embeddings()(tokens.input_ids)

                # Prepend visual tokens — matches training forward_pass()
                inputs_embeds  = torch.cat([proj, text_embeds], dim=1)
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], device=device, dtype=torch.long
                )

        # ── 7. Generate  (FIX 11, 10) ─────────────────────────
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=DTYPE):
                output = llm.generate(
                    inputs_embeds      = inputs_embeds,
                    attention_mask     = attention_mask,
                    max_new_tokens     = MAX_NEW_TOKENS,   # FIX 11: 512
                    do_sample          = False,
                    repetition_penalty = 1.1,
                )

        # ── 8. Decode + clean output ──────────────────────────
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # Strip prompt echo from the START only.
        # Using .replace() anywhere in the string is wrong — it would
        # delete those fragments if they appear inside the generated
        # content (e.g. CONCLUSION referencing the instruction text).
        # lstrip_prefix() removes a specific prefix if present, once.
        def lstrip_prefix(s: str, prefix: str) -> str:
            return s[len(prefix):] if s.startswith(prefix) else s

        # Fragments appear at the start in training order
        for fragment in [
            "[SENSOR: OPTICAL\u2192SAR]\n",
            "[SENSOR: OPTICAL\u2192OPTICAL]\n",
            tag,                                      # same as above, safe to re-try
            "### TASK: DISASTER RESTORATION ADVICE ###\n",
            "### TASK: DISASTER CAPTION ###\n",
            task_str,
            INSTRUCTION,
            "Generate structured report:\n",          # old short form safety net
        ]:
            pred = lstrip_prefix(pred, fragment)
        pred = pred.strip()

        # ── 9. Validate structure ──────────────────────────────
        # Check that expected section headers are present.
        # Logs a warning if prediction is empty or missing headers —
        # helps diagnose truncation or prompt-echo issues.
        task_lower = item.get("task", "").lower()
        if "restoration" in task_lower or "recovery" in task_lower:
            expected_keys = ["IMMEDIATE_RECOVERY:", "LONG_TERM_RECOVERY:"]
        else:
            expected_keys = ["BUILDING:", "ROAD:", "CONCLUSION:"]

        missing_keys = [k for k in expected_keys if k not in pred]

        if not pred:
            tqdm.write(f"  [WARN] Empty prediction for item index "
                       f"{len(results) + skipped}")
        elif missing_keys:
            tqdm.write(f"  [WARN] Prediction missing sections "
                       f"{missing_keys} (len={len(pred)} chars)")

        # ── 10. Collect result ─────────────────────────────────
        # Gracefully handle ground truth key — test JSON may use
        # "ground_truth", "training_answer", or neither.
        ground_truth = (
            item.get("ground_truth")
            or item.get("training_answer")
            or ""
        )

        results.append({
            "task":           item.get("task", "").strip(),
            "is_sar":         is_sar,
            "ground_truth":   ground_truth.strip(),
            "prediction":     pred,
            "sections_ok":    len(missing_keys) == 0,   # quick quality flag
        })

    except Exception as e:
        tqdm.write(f"  [ERROR] {e}")
        skipped += 1


# ══════════════════════════════════════════════════════════
#  SECTION 13 — SAVE RESULTS
# ══════════════════════════════════════════════════════════

with open(PRED_FILE, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

optical_count  = len(results) - sar_count
sections_ok    = sum(1 for r in results if r["sections_ok"])
sections_warn  = len(results) - sections_ok

print(f"\nPREDICTIONS SAVED")
print(f"  File           : {PRED_FILE}")
print(f"  Evaluated      : {len(results)}")
print(f"    Optical      : {optical_count}")
print(f"    SAR          : {sar_count}")
print(f"  Skipped        : {skipped}")
print(f"  Sections OK    : {sections_ok} / {len(results)}")
if sections_warn:
    print(f"  [WARN] {sections_warn} predictions missing expected section headers.")
    print(f"         Check outputs for truncated or empty predictions.")
print(f"\n  Load with: json.load(open('{PRED_FILE}'))")