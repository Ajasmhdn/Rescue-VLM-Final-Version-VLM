import sys, os, json, math, time, signal, traceback, threading
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import open_clip
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from dataset_loader import DisasterDataset
from models.projector_v3 import MultimodalProjectorV3


# ══════════════════════════════════════════════════════════
#  SECTION 1 — CONFIG
# ══════════════════════════════════════════════════════════

TRAIN_JSON          = "data/train.json"
VAL_JSON            = "data/validation.json"
OUTPUT_DIR          = "outputs/multimodal_v3"
CHECKPOINT_DIR      = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_CKPT_DIR       = os.path.join(OUTPUT_DIR, "best_checkpoint")

BATCH_SIZE          = 16
GRAD_ACCUM_STEPS    = 4          # effective batch = 64
EPOCHS              = 12
SAVE_EVERY_N_EPOCHS = 3
EARLY_STOP_PATIENCE = 4

LR_LLM              = 2e-4
LR_CLIP_UNFREEZE    = 5e-6
MAX_GRAD_NORM       = 1.0
WARMUP_RATIO        = 0.05
MAX_SEQ_LEN         = 768
DTYPE               = torch.bfloat16
UNFREEZE_BLOCKS     = [8, 9, 10, 11]

# ── DataLoader — GB10 unified memory safe ──────────────────
NUM_WORKERS         = 2      # ARM bus saturation starts at 4+
PIN_MEMORY          = False  # MUST False: no page-lock on unified mem
PERSISTENT_WORKERS  = False  # MUST False: prevents epoch-boundary hang

# ── Watchdog ───────────────────────────────────────────────
BATCH_TIMEOUT_SEC   = 300
NAN_ABORT_LIMIT     = 3

# Prompt instruction — sensor_tag prepended per-sample in forward_pass
INSTRUCTION = (
    "Carefully analyze differences between pre- and post-disaster images. "
    "Detect even minor structural or environmental changes. "
    "Generate structured report:\n"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Config] Device={device}  Dtype={DTYPE}")
print(f"[Config] Effective batch={BATCH_SIZE}x{GRAD_ACCUM_STEPS}={BATCH_SIZE*GRAD_ACCUM_STEPS}")
print(f"[Config] BEST_CKPT_DIR={BEST_CKPT_DIR}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(BEST_CKPT_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════
#  SECTION 2 — WATCHDOG  (HANG-F)
# ══════════════════════════════════════════════════════════

_watchdog_timer = None

def _watchdog_trip():
    print("\n" + "="*60)
    print(f"WATCHDOG TRIPPED — batch exceeded {BATCH_TIMEOUT_SEC}s")
    print(f"  CUDA allocated : {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  CUDA reserved  : {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print("Stack trace:")
    import faulthandler; faulthandler.dump_traceback()
    print("="*60)
    os.kill(os.getpid(), signal.SIGTERM)

def watchdog_reset():
    global _watchdog_timer
    if _watchdog_timer is not None:
        _watchdog_timer.cancel()
    _watchdog_timer = threading.Timer(BATCH_TIMEOUT_SEC, _watchdog_trip)
    _watchdog_timer.daemon = True
    _watchdog_timer.start()

def watchdog_stop():
    global _watchdog_timer
    if _watchdog_timer is not None:
        _watchdog_timer.cancel()
        _watchdog_timer = None


# ══════════════════════════════════════════════════════════
#  SECTION 3 — SAFE COLLATE  (HANG-H)
#
#  Validates 5-tuple: (pre, post, label, task, is_sar).
#  Matches DisasterDataset.__getitem__ exactly.
#  is_sar bool → BoolTensor(B,) via default_collate.
# ══════════════════════════════════════════════════════════

_bad_sample_log = os.path.join(OUTPUT_DIR, "bad_samples.txt")

def safe_collate(batch):
    clean = []
    for item in batch:
        try:
            if item is None:
                raise ValueError("None sample")
            if len(item) != 5:
                raise ValueError(f"Expected 5-tuple, got {len(item)}")
            pre, post, label, task, is_sar = item
            if not isinstance(pre,  torch.Tensor) or pre.ndim  != 3:
                raise ValueError(f"pre bad shape {getattr(pre,'shape','?')}")
            if not isinstance(post, torch.Tensor) or post.ndim != 3:
                raise ValueError(f"post bad shape {getattr(post,'shape','?')}")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("empty label")
            if not isinstance(task,  str) or not task.strip():
                raise ValueError("empty task")
            if not isinstance(is_sar, bool):
                raise ValueError(f"is_sar not bool: {type(is_sar)}")
            clean.append(item)
        except Exception as e:
            with open(_bad_sample_log, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} | Skipped: {e}\n")
    if not clean:
        return None
    try:
        return default_collate(clean)
    except Exception as e:
        with open(_bad_sample_log, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} | Collate fail: {e}\n")
        return None


# ══════════════════════════════════════════════════════════
#  SECTION 4 — CHECKPOINT HELPERS  (HANG-E + HANG-I)
# ══════════════════════════════════════════════════════════

def get_latest_checkpoint():
    """
    Returns the highest-numbered complete checkpoint.
    Only accepts dirs with ALL 4 required files.
    _tmp dirs from crashed saves are skipped.
    """
    ckpt_epochs = []
    if os.path.isdir(CHECKPOINT_DIR):
        for name in os.listdir(CHECKPOINT_DIR):
            full = os.path.join(CHECKPOINT_DIR, name)
            if name.endswith("_tmp"):
                print(f"  [Resume] Ignoring incomplete save: {name}")
                continue
            if name.startswith("epoch_") and os.path.isdir(full):
                required = ["lora", "projector_v3.pt", "optimizer.pt", "scheduler.pt"]
                if all(os.path.exists(os.path.join(full, r)) for r in required):
                    try:
                        ckpt_epochs.append(int(name.split("_")[1]))
                    except ValueError:
                        pass
                else:
                    missing = [r for r in required
                               if not os.path.exists(os.path.join(full, r))]
                    print(f"  [Resume] Skipping incomplete {name} (missing: {missing})")
    if not ckpt_epochs:
        return None, 0
    latest = max(ckpt_epochs)
    return os.path.join(CHECKPOINT_DIR, f"epoch_{latest}"), latest


def save_checkpoint(epoch, model, projector, clip_model, clip_trainable_params,
                    optimizer, scheduler, logs, subdir=None):
    """Atomic save: write to _tmp, then rename. Crash never corrupts previous ckpt."""
    import shutil
    final_path = subdir if subdir else os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}")
    tmp_path   = final_path + "_tmp"

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)

    try:
        model.save_pretrained(os.path.join(tmp_path, "lora"))
        torch.save(projector.state_dict(),
                   os.path.join(tmp_path, "projector_v3.pt"))
        clip_state = {
            k: v for k, v in clip_model.state_dict().items()
            if any(f"resblocks.{i}" in k for i in UNFREEZE_BLOCKS)
            or "ln_post" in k
        }
        torch.save(clip_state,             os.path.join(tmp_path, "clip_partial.pt"))
        torch.save(optimizer.state_dict(), os.path.join(tmp_path, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(tmp_path, "scheduler.pt"))
        with open(os.path.join(tmp_path, "train_log.json"), "w") as f:
            json.dump(logs, f, indent=4)

        # Atomic replace
        if os.path.exists(final_path):
            shutil.rmtree(final_path)
        os.rename(tmp_path, final_path)
        label = "BEST" if subdir else f"epoch {epoch}"
        tqdm.write(f"  [Checkpoint] Saved {label} -> {final_path}")
    except Exception as e:
        tqdm.write(f"  [Checkpoint] FAILED: {e}")
        traceback.print_exc()


def load_checkpoint_state(ckpt_path, projector, clip_model,
                           optimizer, scheduler, total_steps):
    """
    Loads projector, CLIP partial, optimizer, scheduler.
    LoRA is NOT loaded here — it was already loaded in Section 11
    via PeftModel.from_pretrained. Loading again would double-wrap.
    Returns the existing train_log list for appending.
    """
    tqdm.write(f"\n[Resume] Loading state: {ckpt_path}")
    tqdm.write("  LoRA: loaded earlier via PeftModel.from_pretrained")

    projector.load_state_dict(torch.load(
        os.path.join(ckpt_path, "projector_v3.pt"),
        map_location=device, weights_only=True))
    tqdm.write("  Projector loaded")

    clip_ckpt = os.path.join(ckpt_path, "clip_partial.pt")
    if os.path.exists(clip_ckpt):
        missing, unexpected = clip_model.load_state_dict(
            torch.load(clip_ckpt, map_location=device, weights_only=True),
            strict=False)
        tqdm.write(f"  CLIP loaded (missing={len(missing)} extra={len(unexpected)})")

    optimizer.load_state_dict(torch.load(
        os.path.join(ckpt_path, "optimizer.pt"),
        map_location=device, weights_only=True))
    tqdm.write("  Optimizer loaded")

    sched_state = torch.load(
        os.path.join(ckpt_path, "scheduler.pt"),
        map_location=device, weights_only=True)
    saved_last = sched_state.get("last_epoch", 0)
    if saved_last > total_steps:
        tqdm.write(f"  [WARNING] Scheduler last_epoch={saved_last} > "
                   f"total_steps={total_steps}. Resetting to prevent LR=0.")
    else:
        scheduler.load_state_dict(sched_state)
        tqdm.write(f"  Scheduler loaded (last_epoch={saved_last})")

    log_ckpt = os.path.join(ckpt_path, "train_log.json")
    if os.path.exists(log_ckpt):
        with open(log_ckpt) as f:
            logs = json.load(f)
        tqdm.write(f"  Train log loaded — {len(logs)} epoch(s) recorded:")
        for entry in logs:
            tqdm.write(f"    epoch {entry['epoch']:02d} | "
                       f"train={entry['train_loss']:.4f} | "
                       f"val={entry['val_loss']:.4f}")
        return logs
    tqdm.write("  No train_log.json — fresh log")
    return []


# ══════════════════════════════════════════════════════════
#  SECTION 5 — SENSOR TAG
#
#  Byte-identical to sensor_tag() in test_pipeline_v4.py.
#  First token of every training prompt.
# ══════════════════════════════════════════════════════════

def sensor_tag(is_sar: bool) -> str:
    if is_sar:
        return "[SENSOR: OPTICAL\u2192SAR]\n"
    return "[SENSOR: OPTICAL\u2192OPTICAL]\n"


# ══════════════════════════════════════════════════════════
#  SECTION 6 — TASK PROMPT
#
#  Byte-identical to map_task() in test_pipeline_v4.py.
# ══════════════════════════════════════════════════════════

def map_task(t):
    t = t.lower().strip()
    if t == "disaster caption":
        return "### TASK: DISASTER CAPTION ###\n"
    elif t == "disaster restoration advice":
        return "### TASK: DISASTER RESTORATION ADVICE ###\n"
    else:
        raise ValueError(f"Unknown task: '{t}'")


# ══════════════════════════════════════════════════════════
#  SECTION 7 — CLIP  (HANG-J)
#  float32 intentional — avoids subnormal-value stall on GB10
# ══════════════════════════════════════════════════════════

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="openai"
)
clip_model = clip_model.float().to(device)

for p in clip_model.parameters():
    p.requires_grad = False

clip_trainable_params = []
for name, param in clip_model.named_parameters():
    if (any(f"visual.transformer.resblocks.{i}" in name for i in UNFREEZE_BLOCKS)
            or "visual.ln_post" in name):
        param.requires_grad = True
        clip_trainable_params.append(param)

clip_trainable_m = sum(p.numel() for p in clip_trainable_params) / 1e6
clip_total_m     = sum(p.numel() for p in clip_model.parameters()) / 1e6
print(f"[CLIP] Trainable: {clip_trainable_m:.2f}M / {clip_total_m:.2f}M (float32)")


# ══════════════════════════════════════════════════════════
#  SECTION 8 — DATA  (HANG-A + HANG-H)
# ══════════════════════════════════════════════════════════

train_dataset = DisasterDataset(TRAIN_JSON, preprocess, image_root="data")
val_dataset   = DisasterDataset(VAL_JSON,   preprocess, image_root="data")
print(f"[Data] Train: {len(train_dataset)} | Val: {len(val_dataset)}")

_loader_kwargs = dict(
    batch_size         = BATCH_SIZE,
    num_workers        = NUM_WORKERS,
    pin_memory         = PIN_MEMORY,
    persistent_workers = PERSISTENT_WORKERS,
    collate_fn         = safe_collate,
    prefetch_factor    = 2 if NUM_WORKERS > 0 else None,
    drop_last          = False,
)

train_loader = DataLoader(train_dataset, shuffle=True,  **_loader_kwargs)
val_loader   = DataLoader(val_dataset,   shuffle=False, **_loader_kwargs)


# ══════════════════════════════════════════════════════════
#  SECTION 9 — PROJECTOR
# ══════════════════════════════════════════════════════════

projector = MultimodalProjectorV3().to(device).to(DTYPE)


# ══════════════════════════════════════════════════════════
#  SECTION 10 — TOKENIZER
# ══════════════════════════════════════════════════════════

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token


# ══════════════════════════════════════════════════════════
#  SECTION 11 — MODEL BUILD + AUTO-RESUME
#
#  RESUME PATH:
#    1. Detect latest complete checkpoint
#    2. Build raw base LLM (no LoRA yet)
#    3. Load LoRA via PeftModel.from_pretrained(is_trainable=True)
#       — one clean operation, no double-wrap possible
#    4. Build optimizer + scheduler
#    5. load_checkpoint_state() — projector, CLIP, opt, sched
#    6. Restore best_val_loss + patience from log
#
#  FRESH START PATH:
#    1. Build raw base LLM
#    2. Wrap with get_peft_model() + LoraConfig
#    3. Build optimizer + scheduler
# ══════════════════════════════════════════════════════════

ckpt_path, start_epoch = get_latest_checkpoint()

lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

if ckpt_path is not None:
    print(f"\n[Resume] Checkpoint: {ckpt_path}  (epoch {start_epoch})")
    print(f"[Resume] Will train epochs {start_epoch+1} → {EPOCHS}")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct", torch_dtype=DTYPE,
    ).to(device)

    # Single clean load — no double-adapter wrap
    model = PeftModel.from_pretrained(
        model,
        os.path.join(ckpt_path, "lora"),
        is_trainable=True,
    )
    model.print_trainable_parameters()

else:
    print("\n[Fresh] No checkpoint — starting from epoch 1")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct", torch_dtype=DTYPE,
    ).to(device)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


# ══════════════════════════════════════════════════════════
#  SECTION 12 — OPTIMIZER + SCHEDULER
# ══════════════════════════════════════════════════════════

total_steps  = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

def build_optimizer():
    return torch.optim.AdamW([
        {"params": model.parameters(),     "lr": LR_LLM,           "weight_decay": 0.01},
        {"params": projector.parameters(), "lr": LR_LLM,           "weight_decay": 0.01},
        {"params": clip_trainable_params,  "lr": LR_CLIP_UNFREEZE, "weight_decay": 0.0},
    ])

def build_scheduler(opt):
    return get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

optimizer = build_optimizer()
scheduler = build_scheduler(optimizer)
print(f"[Scheduler] total_steps={total_steps}  warmup={warmup_steps}")


# ══════════════════════════════════════════════════════════
#  SECTION 13 — COMPLETE RESUME / INIT TRAINING STATE
# ══════════════════════════════════════════════════════════

best_val_loss    = float("inf")
patience_counter = 0

if ckpt_path is not None:
    logs = load_checkpoint_state(
        ckpt_path, projector, clip_model,
        optimizer, scheduler, total_steps,
    )
    if logs:
        best_val_loss = min(e["val_loss"] for e in logs)
        # Recompute patience: trailing epochs without improvement
        patience_counter = 0
        for entry in reversed(logs):
            if entry.get("val_loss", float("inf")) > best_val_loss:
                patience_counter += 1
            else:
                break
        tqdm.write(f"\n[Resume] best_val_loss={best_val_loss:.4f}  "
                   f"patience={patience_counter}/{EARLY_STOP_PATIENCE}")
    print(f"[Resume] train_log.json will APPEND from epoch {start_epoch+1}\n")
else:
    logs = []
    print()


# ══════════════════════════════════════════════════════════
#  SECTION 14 — FUSION HELPER
# ══════════════════════════════════════════════════════════

def fuse_features(pre_feat, post_feat):
    diff = post_feat - pre_feat
    return torch.cat([pre_feat, post_feat, diff], dim=1)   # (B, 1536)


# ══════════════════════════════════════════════════════════
#  SECTION 15 — FORWARD PASS
#
#  Prompt per sample: sensor_tag + map_task + INSTRUCTION + label
#  Byte-identical to test_pipeline_v4.py prompt construction.
#
#  is_sar_flags: BoolTensor (B,) from DataLoader.
#  Stays on CPU — only used for string formatting, not GPU math.
# ══════════════════════════════════════════════════════════

def forward_pass(pre_img, post_img, labels, tasks, is_sar_flags):
    # CLIP in float32 — no autocast here (HANG-J)
    pre_feat  = clip_model.encode_image(pre_img.float())
    post_feat = clip_model.encode_image(post_img.float())
    fusion    = fuse_features(pre_feat, post_feat)   # (B, 1536) float32

    with torch.amp.autocast("cuda", dtype=DTYPE):
        proj = projector(fusion)   # (B, 64, 1536) bfloat16

        texts = [
            sensor_tag(bool(sar)) + map_task(t) + INSTRUCTION + l
            for sar, t, l in zip(is_sar_flags, tasks, labels)
        ]
        tokens = tokenizer(
            texts, padding=True, truncation=True,
            max_length=MAX_SEQ_LEN, return_tensors="pt",
        ).to(device)

        text_embeds = model.get_input_embeddings()(tokens.input_ids)
        inputs      = torch.cat([proj, text_embeds], dim=1)
        attn_mask   = torch.ones(inputs.shape[:2], device=device, dtype=torch.long)

        label_ids = tokens.input_ids.clone()
        label_ids[label_ids == tokenizer.pad_token_id] = -100
        img_mask  = torch.full(
            (label_ids.shape[0], proj.shape[1]),
            -100, device=device, dtype=torch.long,
        )
        new_labels = torch.cat([img_mask, label_ids], dim=1)

        outputs = model(
            inputs_embeds=inputs, attention_mask=attn_mask, labels=new_labels,
        )
    return outputs.loss


# ══════════════════════════════════════════════════════════
#  SECTION 16 — SANITY CHECK  (fresh start only)
# ══════════════════════════════════════════════════════════

if start_epoch == 0:
    print("[Sanity] Fresh start — running 30-step overfit check...")
    clip_model.train(); model.train(); projector.train()

    sb = next(iter(train_loader))
    if sb is None:
        print("[Sanity] WARNING: first batch None — check dataset")
    else:
        pre_s    = sb[0][:1].to(device)
        post_s   = sb[1][:1].to(device)
        lab_s    = [sb[2][0]]
        task_s   = [sb[3][0]]
        is_sar_s = sb[4][:1]   # BoolTensor (1,)

        sanity_opt = torch.optim.AdamW(
            list(model.parameters()) + list(projector.parameters()), lr=1e-3)

        sl = None
        for step in range(30):
            sl = forward_pass(pre_s, post_s, lab_s, task_s, is_sar_s)
            if not torch.isfinite(sl):
                print(f"  NaN/Inf at step {step} — check model dims!"); break
            sanity_opt.zero_grad()
            sl.backward()
            sanity_opt.step()
            if step % 10 == 0:
                print(f"  Step {step:02d} | loss={sl.item():.4f}")

        if sl is not None:
            status = "OK" if sl.item() <= 2.0 else "HIGH — check dims!"
            print(f"  Final sanity loss: {sl.item():.4f}  [{status}]")

    alloc  = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved()  / 1e9
    print(f"[Memory] Alloc={alloc:.1f}GB  Reserved={reserv:.1f}GB  "
          f"Free~{130.7-reserv:.1f}GB\n")

    torch.cuda.empty_cache()
    optimizer = build_optimizer()
    scheduler = build_scheduler(optimizer)

else:
    print(f"[Sanity] Skipped — resuming from epoch {start_epoch}.\n")


# ══════════════════════════════════════════════════════════
#  SECTION 17 — TRAINING LOOP
#
#  5-element batch: pre_img, post_img, labels, tasks, is_sar_flags
#  is_sar_flags: BoolTensor(B,) — passed directly to forward_pass.
# ══════════════════════════════════════════════════════════

for epoch in range(start_epoch, EPOCHS):
    tqdm.write(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ── TRAIN ──────────────────────────────────────────────
    clip_model.train(); model.train(); projector.train()

    total_train_loss  = 0.0
    accum_loss_scalar = 0.0
    skipped_batches   = 0
    consecutive_nan   = 0
    optimizer.zero_grad()

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1} train", dynamic_ncols=True)

    for step_i, batch in enumerate(loop):

        watchdog_reset()

        if batch is None:
            skipped_batches += 1
            continue

        pre_img, post_img, labels, tasks, is_sar_flags = batch
        pre_img  = pre_img.to(device,  non_blocking=False)
        post_img = post_img.to(device, non_blocking=False)
        # is_sar_flags stays on CPU — string formatting only

        try:
            loss = forward_pass(
                pre_img, post_img,
                list(labels), list(tasks),
                is_sar_flags,
            )
        except RuntimeError as e:
            tqdm.write(f"  [ERR] step {step_i}: {e}")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            skipped_batches += 1
            continue

        if not torch.isfinite(loss):
            consecutive_nan += 1
            tqdm.write(f"  [NaN] step {step_i}  ({consecutive_nan}/{NAN_ABORT_LIMIT})")
            optimizer.zero_grad()
            if consecutive_nan >= NAN_ABORT_LIMIT:
                tqdm.write("  [ABORT] Saving checkpoint and exiting.")
                save_checkpoint(epoch+1, model, projector, clip_model,
                                clip_trainable_params, optimizer, scheduler, logs)
                watchdog_stop()
                sys.exit(1)
            continue
        else:
            consecutive_nan = 0

        scaled = loss / GRAD_ACCUM_STEPS
        scaled.backward()
        accum_loss_scalar += scaled.item()
        total_train_loss  += loss.item()

        if (step_i + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(projector.parameters())
                + clip_trainable_params, MAX_GRAD_NORM)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            accum_loss_scalar = 0.0

        loop.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
            skip=skipped_batches,
        )

    # Flush remaining accumulated gradients
    if len(train_loader) % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(projector.parameters())
            + clip_trainable_params, MAX_GRAD_NORM)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

    effective  = max(1, len(train_loader) - skipped_batches)
    train_loss = total_train_loss / effective
    if skipped_batches:
        tqdm.write(f"  [Info] Skipped {skipped_batches} batches")

    torch.cuda.empty_cache()

    # ── VALIDATION ──────────────────────────────────────────
    clip_model.eval(); model.eval(); projector.eval()

    total_val_loss = 0.0
    val_batches    = 0
    val_skipped    = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="val", dynamic_ncols=True):
            watchdog_reset()
            if batch is None:
                val_skipped += 1; continue
            pre_img, post_img, labels, tasks, is_sar_flags = batch
            pre_img  = pre_img.to(device)
            post_img = post_img.to(device)
            try:
                v = forward_pass(
                    pre_img, post_img,
                    list(labels), list(tasks),
                    is_sar_flags,
                )
                if torch.isfinite(v):
                    total_val_loss += v.item()
                    val_batches    += 1
            except RuntimeError as e:
                tqdm.write(f"  [Val ERR] {e}")
                val_skipped += 1

    val_loss = total_val_loss / max(1, val_batches)
    torch.cuda.empty_cache()

    improved = val_loss < best_val_loss
    marker   = " <- best" if improved else ""
    tqdm.write(
        f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{marker}"
        + (f"  (val_skip={val_skipped})" if val_skipped else "")
    )

    # Append to existing log — no gaps, no overwrites
    logs.append({
        "epoch":      epoch + 1,
        "train_loss": round(train_loss, 6),
        "val_loss":   round(val_loss,   6),
        "skipped":    skipped_batches,
    })

    if improved:
        best_val_loss    = val_loss
        patience_counter = 0
        save_checkpoint(epoch+1, model, projector, clip_model,
                        clip_trainable_params, optimizer, scheduler, logs,
                        subdir=BEST_CKPT_DIR)
        tqdm.write(f"  New best val loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        tqdm.write(f"  No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

    if (epoch+1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch+1) == EPOCHS:
        save_checkpoint(epoch+1, model, projector, clip_model,
                        clip_trainable_params, optimizer, scheduler, logs)

    if patience_counter >= EARLY_STOP_PATIENCE:
        tqdm.write(f"\nEarly stopping at epoch {epoch+1}. "
                   f"Best val loss: {best_val_loss:.4f}")
        break


# ══════════════════════════════════════════════════════════
#  SECTION 18 — FINAL SAVE
# ══════════════════════════════════════════════════════════

watchdog_stop()

tqdm.write("\n[Final] Saving last epoch weights...")
try:
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora"))
    torch.save(projector.state_dict(),
               os.path.join(OUTPUT_DIR, "projector_v3.pt"))
    torch.save(
        {k: v for k, v in clip_model.state_dict().items()
         if any(f"resblocks.{i}" in k for i in UNFREEZE_BLOCKS) or "ln_post" in k},
        os.path.join(OUTPUT_DIR, "clip_partial_ft.pt"),
    )
    with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w") as f:
        json.dump(logs, f, indent=4)
except Exception as e:
    tqdm.write(f"[Final Save ERROR] {e}")
    tqdm.write(f"Best checkpoint intact at: {BEST_CKPT_DIR}")

tqdm.write(f"\nTRAINING COMPLETE")
tqdm.write(f"  Best val loss : {best_val_loss:.4f}")
tqdm.write(f"  Best model    : {BEST_CKPT_DIR}")
tqdm.write(f"  Last model    : {OUTPUT_DIR}/lora")
tqdm.write(f"  Total epochs  : {len(logs)}")
tqdm.write("\n  Full training log:")
for entry in logs:
    tqdm.write(f"    epoch {entry['epoch']:02d} | "
               f"train={entry['train_loss']:.4f} | "
               f"val={entry['val_loss']:.4f}")

bad_log_size = os.path.getsize(_bad_sample_log) if os.path.exists(_bad_sample_log) else 0
if bad_log_size > 0:
    tqdm.write(f"\n  WARNING: corrupt samples logged to {_bad_sample_log}")
tqdm.write(f"\n  For inference, always load from: {BEST_CKPT_DIR}")
