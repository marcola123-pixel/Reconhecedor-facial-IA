"""Carregar um experimento treinado e gerar embeddings (usado pela aplicação e pelos scripts)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from .data import build_transform
from .models import build_model


def load_run(run_dir: str | Path, device: str = "cpu") -> Tuple[torch.nn.Module, dict, object]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    model = build_model(meta["model"], meta["image_size"], load_pretrained=False)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state, strict = False)
    model.to(device).eval()
    return model, meta, build_transform(meta["image_size"], train=False)


@torch.no_grad()
def embed_images(model, transform, images: List[Image.Image], device: str = "cpu") -> np.ndarray:
    x = torch.stack([transform(im.convert("RGB")) for im in images]).to(device)
    return model.eval()(x).cpu().numpy().astype(np.float64)
