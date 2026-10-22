import sys
import os
import numpy as np
import torch
import open_clip
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image, ImageOps, ImageFilter

# ──────────────────────────────────────────────────────────
#  PATH SETUP
# ──────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from models.projector_v3 import MultimodalProjectorV3

# ──────────────────────────────────────────────────────────
#  CONFIG — edit these lines for your test images
# ──────────────────────────────────────────────────────────

device   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16

CKPT_DIR = os.path.join(BASE_DIR, "outputs/multimodal_v3/best_checkpoint")

PRE_IMAGE  = os.path.join(BASE_DIR, "bata_explosion_pre_0.png")
POST_IMAGE = os.path.join(BASE_DIR, "bata_explosion_post_0.png")

# True if post image is SAR, False if optical
IS_SAR_POST = True    # ← change to False for optical post images

TASK = "recovery"     # "caption" or "recovery"

UNFREEZE_BLOCKS = [8, 9, 10, 11]


# ──────────────────────────────────────────────────────────
#  CONSTANTS — copied verbatim from train_vlm_v3_dgx.py
#  DO NOT edit these. They must match training exactly.
# ──────────────────────────────────────────────────────────

INSTRUCTION = (
    "Carefully analyze differences between pre- and post-disaster images. "
    "Detect even minor structural or environmental changes. "
    "Generate structured report:\n"
)

# ──────────────────────────────────────────────────────────
#  TOKEN BUDGET — task aware
#
#  Recovery  → 260  (IMMEDIATE + LONG_TERM ~ 220-250 tokens)
#  Caption   → 512  (6 sections fill the full budget naturally)
#
#  DO NOT set both to 512. Recovery has no natural stop section
#  (no CONCLUSION equivalent). Excess budget = bleed + loops.
# ──────────────────────────────────────────────────────────

MAX_NEW_TOKENS = {
    "recovery" : 260,
    "caption"  : 512,
}


# ──────────────────────────────────────────────────────────
#  SENSOR TAG — identical to sensor_tag() in training
#
#  Training code (train_vlm_v3_dgx.py, Section 6):
#    def sensor_tag(is_sar: bool) -> str:
#        if is_sar:
#            return "[SENSOR: OPTICAL\u2192SAR]\n"
#        return "[SENSOR: OPTICAL\u2192OPTICAL]\n"
#
#  The unicode arrow \u2192 is "→". Kept as unicode escape here
#  to guarantee byte-identical output regardless of file encoding.
# ──────────────────────────────────────────────────────────

def sensor_tag(is_sar: bool) -> str:
    if is_sar:
        return "[SENSOR: OPTICAL\u2192SAR]\n"
    return "[SENSOR: OPTICAL\u2192OPTICAL]\n"


# ──────────────────────────────────────────────────────────
#  TASK MAP — identical to map_task() in training
# ──────────────────────────────────────────────────────────

def map_task(t: str) -> str:
    t = t.lower().strip()
    if "restoration" in t or "recovery" in t:
        return "### TASK: DISASTER RESTORATION ADVICE ###\n"
    return "### TASK: DISASTER CAPTION ###\n"


# ──────────────────────────────────────────────────────────
#  SAR PREPROCESSING — identical to preprocess_sar() in
#  dataset_loader.py V4. Three stages, same order.
#
#  Stage 1: MedianFilter(3) — despeckle before any math
#  Stage 2: log1p            — compress log-normal → Gaussian
#  Stage 3: percentile [1,99] stretch on log values
#
#  FIX vs old test: added Stage 1 (was missing), removed the
#  unsharp mask that was in old test but NOT in training.
# ──────────────────────────────────────────────────────────

def preprocess_sar(pil_img: Image.Image) -> Image.Image:
    # Stage 1 — Despeckle (MUST be first, before any intensity math)
    img_gray = ImageOps.grayscale(pil_img)
    img_desp = img_gray.filter(ImageFilter.MedianFilter(size=3))
    arr      = np.array(img_desp, dtype=np.float32)

    # Stage 2 — Log-scale: SAR backscatter is log-normally distributed
    arr = np.log1p(arr)

    # Stage 3 — Percentile stretch on log values
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    # 3-channel grayscale so CLIP preprocess works unchanged
    arr_rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr_rgb, mode="RGB")


# ──────────────────────────────────────────────────────────
#  STARTUP BANNER
# ──────────────────────────────────────────────────────────

print("=" * 55)
print("  TEST PIPELINE V4 — SAR MODALITY AWARE")
print("=" * 55)
print(f"  Device    : {device}")
print(f"  Dtype     : {DTYPE}")
print(f"  Ckpt dir  : {CKPT_DIR}")
print(f"  Post type : {'SAR' if IS_SAR_POST else 'Optical'}")
print(f"  Task      : {map_task(TASK).strip()}")
print("=" * 55 + "\n")

# ──────────────────────────────────────────────────────────
#  CHECKPOINT FILES CHECK
#  Only inference-required files checked.
#  optimizer.pt / scheduler.pt are training artefacts — not
#  loaded at inference, removed from this check (FIX 5).
# ──────────────────────────────────────────────────────────

_required_inference = ["lora", "projector_v3.pt", "clip_partial.pt"]
all_ok = True
print("Checkpoint files:")
for r in _required_inference:
    full   = os.path.join(CKPT_DIR, r)
    exists = os.path.exists(full)
    status = "OK" if exists else "MISSING"
    if not exists:
        all_ok = False
    print(f"  [{status}] {r}")
if not all_ok:
    print("\nERROR: Missing required checkpoint files. Aborting.")
    sys.exit(1)
print()

# ──────────────────────────────────────────────────────────
#  LOAD CLIP — float32, fine-tuned blocks from checkpoint
# ──────────────────────────────────────────────────────────

print("Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="openai"
)
clip_model = clip_model.float().to(device)  # float32 intentional (matches training)

missing, unexpected = clip_model.load_state_dict(
    torch.load(
        os.path.join(CKPT_DIR, "clip_partial.pt"),
        map_location=device,
        weights_only=True,
    ),
    strict=False,
)
clip_model.eval()
print(f"  CLIP fine-tuned blocks loaded  "
      f"(missing={len(missing)}, extra={len(unexpected)})")
print("  CLIP ready (float32)\n")

# ──────────────────────────────────────────────────────────
#  LOAD PROJECTOR
# ──────────────────────────────────────────────────────────

print("Loading Projector V3...")
projector = MultimodalProjectorV3().to(device).to(DTYPE)
projector.load_state_dict(
    torch.load(
        os.path.join(CKPT_DIR, "projector_v3.pt"),
        map_location=device,
        weights_only=True,
    )
)
projector.eval()
print("  Projector V3 ready\n")

# ──────────────────────────────────────────────────────────
#  LOAD LLM + LoRA
# ──────────────────────────────────────────────────────────

print("Loading LLM + LoRA...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=DTYPE,
).to(device)
llm = PeftModel.from_pretrained(llm, os.path.join(CKPT_DIR, "lora"))
llm.eval()
print("  LLM + LoRA ready\n")

# ──────────────────────────────────────────────────────────
#  LOAD + PREPROCESS IMAGES
# ──────────────────────────────────────────────────────────

def load_image(path: str, is_sar: bool) -> torch.Tensor:
    """Load, preprocess, return (1, C, H, W) tensor."""
    with Image.open(path) as img:
        if is_sar:
            img = preprocess_sar(img)
            label = "SAR (3-stage pipeline)"
        else:
            img = img.convert("RGB")
            label = "Optical (RGB)"
        tensor = preprocess(img).unsqueeze(0).to(device)
    print(f"  {label:<30}  {os.path.basename(path)}")
    return tensor

print("Loading images...")
pre_img  = load_image(PRE_IMAGE,  is_sar=False)       # pre always optical
post_img = load_image(POST_IMAGE, is_sar=IS_SAR_POST)
print()

# ──────────────────────────────────────────────────────────
#  CLIP ENCODE — float32, no autocast (matches training)
# ──────────────────────────────────────────────────────────

with torch.no_grad():
    pre_feat  = clip_model.encode_image(pre_img.float())   # (1, 512)
    post_feat = clip_model.encode_image(post_img.float())  # (1, 512)

assert pre_feat.shape  == (1, 512), f"CLIP dim error pre:  {pre_feat.shape}"
assert post_feat.shape == (1, 512), f"CLIP dim error post: {post_feat.shape}"
print(f"CLIP features : {pre_feat.shape} each")

# ──────────────────────────────────────────────────────────
#  FUSION — [pre | post | post-pre] = 1536-dim
#  Identical to fuse_features() in training.
# ──────────────────────────────────────────────────────────

diff        = post_feat - pre_feat
fusion_feat = torch.cat([pre_feat, post_feat, diff], dim=1)  # (1, 1536)

assert fusion_feat.shape == (1, 1536), f"Fusion dim error: {fusion_feat.shape}"
diff_norm = diff.norm().item()
print(f"Fusion shape  : {fusion_feat.shape}  [pre | post | diff]")
print(f"Diff norm     : {diff_norm:.4f}  "
      f"({'structural change signal present' if diff_norm > 1.0 else 'low change — verify images'})\n")

# ──────────────────────────────────────────────────────────
#  PROJECT — float32 in, bfloat16 out (matches training)
# ──────────────────────────────────────────────────────────

with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=DTYPE):
        proj_feat = projector(fusion_feat)   # (1, N_tokens, dim)

print(f"Projected shape: {proj_feat.shape}\n")

# ──────────────────────────────────────────────────────────
#  BUILD PROMPT — identical to training forward_pass()
#
#  Training (train_vlm_v3_dgx.py, forward_pass):
#    texts = [
#        sensor_tag(bool(sar)) + map_task(t) + INSTRUCTION + l
#        for sar, t, l in zip(is_sar_flags, tasks, labels)
#    ]
#
#  At inference there is no ground-truth label (l) — we want
#  the model to generate it. So the prompt is:
#    sensor_tag + map_task + INSTRUCTION
#  and generation continues from there.
# ──────────────────────────────────────────────────────────

tag         = sensor_tag(IS_SAR_POST)
task_str    = map_task(TASK)
full_prompt = tag + task_str + INSTRUCTION

print("Prompt structure:")
print(f"  sensor_tag  : {repr(tag)}")
print(f"  task        : {repr(task_str)}")
print(f"  instruction : {repr(INSTRUCTION[:60])}...")
print()

# ──────────────────────────────────────────────────────────
#  TOKENIZE + EMBED PROMPT
# ──────────────────────────────────────────────────────────

with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=DTYPE):
        tokens      = tokenizer(full_prompt, return_tensors="pt").to(device)
        text_embeds = llm.get_input_embeddings()(tokens.input_ids)

        # Prepend visual tokens — same as training forward_pass()
        inputs_embeds  = torch.cat([proj_feat, text_embeds], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], device=device, dtype=torch.long
        )

print(f"Sequence : {inputs_embeds.shape[1]} tokens total  "
      f"(visual={proj_feat.shape[1]}, text={text_embeds.shape[1]})\n")

# ──────────────────────────────────────────────────────────
#  RESOLVE TOKEN BUDGET
#
#  "recovery" in task_str → 260  (IMMEDIATE + LONG_TERM only)
#  otherwise              → 512  (caption needs 6 sections)
#
#  Why this works:
#    Recovery generates ~220-250 tokens for its two sections.
#    260 cuts generation right after LONG_TERM ends, leaving no
#    room for the instruction-repeat / second-loop / caption-bleed.
#    Caption naturally fills ~450-500 tokens across 6 sections
#    and reaches CONCLUSION before the 512 limit.
# ──────────────────────────────────────────────────────────

task_key   = "recovery" if "recovery" in TASK.lower() or "restoration" in TASK.lower() else "caption"
max_tokens = MAX_NEW_TOKENS[task_key]

print(f"Token budget  : {max_tokens}  ({task_key})\n")

# ──────────────────────────────────────────────────────────
#  GENERATE
# ──────────────────────────────────────────────────────────

print("Generating...\n")
with torch.no_grad():
    output = llm.generate(
        inputs_embeds      = inputs_embeds,
        attention_mask     = attention_mask,
        max_new_tokens     = max_tokens,
        do_sample          = False,
        repetition_penalty = 1.1,
        eos_token_id       = tokenizer.eos_token_id,
    )

# ──────────────────────────────────────────────────────────
#  DECODE + STRIP PROMPT ECHOES
# ──────────────────────────────────────────────────────────
decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()

# ── Strip prompt echoes ────────────────────────────────────
for fragment in [tag, task_str, INSTRUCTION]:
    decoded = decoded.replace(fragment, "")
decoded = decoded.strip()

# ── Recovery: cut at first bleed marker after LONG_TERM ───
if "recovery" in TASK.lower() or "restoration" in TASK.lower():
    if "LONG_TERM_RECOVERY:" in decoded:
        before, after = decoded.split("LONG_TERM_RECOVERY:", 1)
        bleed_markers = [
            "(END)",
            "Carefully analyze",
            "IMMEDIATE_RECOVERY:",
            "### TASK:",
            "[SENSOR:",          # ← sensor tag bleed
            tag,                 # ← exact sensor tag for this run
        ]
        cut_pos = len(after)
        for marker in bleed_markers:
            pos = after.find(marker)
            if pos != -1 and pos < cut_pos:
                cut_pos = pos
        after   = after[:cut_pos].strip()
        decoded = before.strip() + "\nLONG_TERM_RECOVERY: " + after
# ──────────────────────────────────────────────────────────
#  OUTPUT
# ──────────────────────────────────────────────────────────

print("=" * 55)
print(f"  TASK     : {task_str.strip()}")
print(f"  SENSOR   : {'OPTICAL → SAR (radar)' if IS_SAR_POST else 'OPTICAL → OPTICAL'}")
print(f"  CKPT     : best_checkpoint/")
print(f"  BUDGET   : {max_tokens} tokens")
print("=" * 55)
print()
print(decoded)
print()
print("=" * 55)
