# ==========================================================
# DATASET LOADER — CANONICAL V4
#
# SAR pipeline: V4 — 3 stages (matches test_pipeline_v4.py)
#   Stage 1: MedianFilter(3)       despeckle before any math
#   Stage 2: log1p                 compress log-normal → Gaussian
#   Stage 3: percentile [1,99]     stretch on log values
#
#   No unsharp mask — that was a V3 step not present in
#   the active inference pipeline.
#
# Return: 5-tuple (pre_img, post_img, label, task, is_sar)
#   is_sar bool is returned so forward_pass() can prepend the
#   correct sensor_tag per sample. default_collate converts a
#   list of bools into a BoolTensor(B,) automatically.
#
# ALIGNMENT:
#   preprocess_sar() here is byte-identical to preprocess_sar()
#   in test_pipeline_v4.py and generate_test_predictions_v4.py.
#   Any change to one must be applied to all three.
# ==========================================================

import os
import json
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset


def preprocess_sar(pil_img: Image.Image) -> Image.Image:
    """
    V4 SAR preprocessing — 3 stages.

    Stage 1 — MedianFilter(3) despeckle
      SAR has multiplicative speckle noise. Despeckle FIRST,
      before any intensity math, so log1p and percentile stretch
      operate on clean values. MedianFilter is edge-preserving
      (unlike mean blur), keeping building and road boundaries.

    Stage 2 — log1p compression
      Radar backscatter is log-normally distributed (~40 dB range).
      log1p maps it toward the Gaussian distribution that CLIP's
      ViT-B/16 was trained on. log1p(0) = 0, dark pixels safe.

    Stage 3 — percentile stretch [1, 99]
      Clips extreme values (residual outliers after despeckle)
      before normalisation. Applied after log so it operates on
      the compressed dynamic range.

    Output: 3-channel grayscale RGB — CLIP preprocess expects
    (3, H, W). Uniform stack is equivalent to single-channel.
    """
    # Stage 1 — despeckle
    img_gray = ImageOps.grayscale(pil_img)
    img_desp = img_gray.filter(ImageFilter.MedianFilter(size=3))
    arr      = np.array(img_desp, dtype=np.float32)

    # Stage 2 — log compression
    arr = np.log1p(arr)

    # Stage 3 — percentile stretch
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    # Stack to 3-channel RGB
    arr_rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr_rgb, mode="RGB")


class DisasterDataset(Dataset):
    """
    Returns 5-tuple per sample:
        (pre_img, post_img, label, task, is_sar)

    pre_img  : Tensor (3, H, W) — always optical
    post_img : Tensor (3, H, W) — optical or V4-SAR-preprocessed
    label    : str — training_answer
    task     : str — "disaster caption" | "disaster restoration advice"
    is_sar   : bool — True when post image is SAR radar
    """

    def __init__(self, json_path: str, preprocess, image_root: str = None):
        self.preprocess = preprocess
        self.image_root = image_root
        self.samples    = []

        with open(json_path, "r") as f:
            data = json.load(f)

        skipped   = 0
        sar_count = 0

        for item in data:
            try:
                pre  = item["pre_image_path"].replace("\\", "/")
                post = item["post_image_path"].replace("\\", "/")

                if self.image_root:
                    pre  = os.path.normpath(os.path.join(self.image_root, pre))
                    post = os.path.normpath(os.path.join(self.image_root, post))

                if not os.path.exists(pre) or not os.path.exists(post):
                    skipped += 1
                    continue

                label = item.get("training_answer", "").strip()
                if not label:
                    skipped += 1
                    continue

                task_raw = item.get("task", "").strip().lower()
                if "restoration" in task_raw or "recovery" in task_raw:
                    task = "disaster restoration advice"
                else:
                    task = "disaster caption"

                post_type = item.get("post_image_type", "Optical").strip().upper()
                is_sar    = (post_type == "SAR")
                if is_sar:
                    sar_count += 1

                self.samples.append({
                    "pre":    pre,
                    "post":   post,
                    "label":  label,
                    "task":   task,
                    "is_sar": is_sar,
                })

            except Exception:
                skipped += 1

        opt_count = len(self.samples) - sar_count
        print(f"[Dataset] Loaded: {len(self.samples)} "
              f"(optical={opt_count}, SAR={sar_count}) | Skipped: {skipped}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str, is_sar: bool):
        with Image.open(path) as img:
            if is_sar:
                img = preprocess_sar(img)   # V4 3-stage
            else:
                img = img.convert("RGB")
            return self.preprocess(img)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        try:
            pre_img  = self._load_image(s["pre"],  is_sar=False)
            post_img = self._load_image(s["post"], is_sar=s["is_sar"])
            return pre_img, post_img, s["label"], s["task"], s["is_sar"]
        except Exception:
            # Adjacent-sample fallback, no recursion
            fb = self.samples[(idx + 1) % len(self.samples)]
            pre_img  = self._load_image(fb["pre"],  is_sar=False)
            post_img = self._load_image(fb["post"], is_sar=fb["is_sar"])
            return pre_img, post_img, fb["label"], fb["task"], fb["is_sar"]