"""Dataset, transformações e amostrador P×K (precisa de PyTorch)."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .splits import Identities


def build_transform(image_size: int, train: bool = False):
    """Mesma transformação em treino/avaliação/aplicação (só muda a augmentation do treino)."""
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)]
    ops += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    return transforms.Compose(ops)


class FaceDataset(Dataset):
    """Rostos já recortados, organizados como <identidade>/<foto>. Rótulo = índice da identidade."""

    def __init__(self, identities: Identities, transform=None):
        self.class_names = sorted(identities)
        self.samples = [(p, ci) for ci, n in enumerate(self.class_names) for p in identities[n]]
        self.labels = np.array([c for _, c in self.samples], dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), label


class PKBatchSampler(Sampler):
    """Cada batch = P identidades × K fotos (com reposição se a pessoa tiver < K fotos).
    Garante positivos e negativos em todo batch — requisito prático de triplet/contrastive/InfoNCE."""

    def __init__(self, labels: Sequence[int], P: int, K: int, steps: int, seed: int = 0):
        self.by_id: dict[int, List[int]] = {}
        for i, l in enumerate(labels):
            self.by_id.setdefault(int(l), []).append(i)
        self.ids = sorted(self.by_id)
        self.P, self.K, self.steps = min(P, len(self.ids)), K, steps
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            chosen = self.rng.choice(self.ids, size=self.P, replace=False)
            batch: List[int] = []
            for l in chosen:
                pool = self.by_id[int(l)]
                batch += [int(i) for i in self.rng.choice(pool, size=self.K, replace=len(pool) < self.K)]
            yield batch
