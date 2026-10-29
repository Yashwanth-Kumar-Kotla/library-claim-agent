"""Metric depth from one image with Apple's Depth Pro (pretrained, Apple sample-code licence).

Returns depth in metres per pixel and the model's own focal-length estimate, so no camera
metadata is needed. Measured on an M5 MacBook: ~2.3 s per 1920x1080 frame once loaded.
One model instance is shared and calls are serialised: the GPU runs one frame at a time anyway.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

MODEL_ID = "apple/DepthPro-hf"


class DepthPro:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._model = None
        self._proc = None
        self._lock = threading.Lock()

    def load(self) -> "DepthPro":
        with self._lock:
            if self._model is None:
                import torch
                from transformers import DepthProForDepthEstimation, DepthProImageProcessor
                device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device != "cpu" else torch.float32
                self._proc = DepthProImageProcessor.from_pretrained(self.model_id)
                self._model = DepthProForDepthEstimation.from_pretrained(self.model_id, dtype=dtype).to(device).eval()
                self._device, self._dtype = device, dtype
        return self

    def warm_up(self) -> None:
        """Load in the background at server start so the first sweep doesn't wait for it."""
        threading.Thread(target=self.load, daemon=True).start()

    def __call__(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        import torch
        self.load()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            inputs = self._proc(images=rgb, return_tensors="pt").to(self._device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
            with torch.no_grad():
                out = self._model(**inputs)
            post = self._proc.post_process_depth_estimation(out, target_sizes=[rgb.shape[:2]])[0]
        depth = post["predicted_depth"].float().cpu().numpy()
        focal = float(post["focal_length"])
        return depth, focal
