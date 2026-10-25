"""Separação dos dados e protocolos de avaliação (somente numpy/stdlib).

Regra central (spec 006): a separação treino/validação/teste é feita **por identidade**.
Assim, as pessoas do teste nunca aparecem no treino — o que simula "pessoa nova" e evita
data leakage entre imagens da mesma pessoa.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
Identities = Dict[str, List[str]]          # identidade -> lista de caminhos
Split = Dict[str, Identities]              # "train"/"val"/"test" -> Identities


def scan_identities(root: str | Path, min_images: int = 2) -> Identities:
    """Lê `root/<identidade>/*.img` e mantém identidades com pelo menos `min_images` fotos."""
    out: Identities = {}
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        files = sorted(str(f) for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        if len(files) >= min_images:
            out[d.name] = files
    return out


def split_identities(identities: Identities, fractions=(0.7, 0.15, 0.15), seed: int = 42) -> Split:
    """Divide as IDENTIDADES (não as imagens) em train/val/test, de forma reprodutível."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("as frações precisam somar 1")
    names = sorted(identities)
    random.Random(seed).shuffle(names)
    n = len(names)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    parts = {
        "train": names[:n_train],
        "val": names[n_train:n_train + n_val],
        "test": names[n_train + n_val:],
    }
    for part, ids in parts.items():
        if len(ids) < 2:
            raise ValueError(f"partição '{part}' ficou com {len(ids)} identidade(s); "
                             f"há poucas identidades ({n}) para essas frações")
    return {k: {name: identities[name] for name in v} for k, v in parts.items()}


def assert_disjoint_identities(split: Split) -> None:
    seen: Dict[str, str] = {}
    for part, ids in split.items():
        for name in ids:
            if name in seen:
                raise AssertionError(f"identidade '{name}' aparece em '{seen[name]}' e em '{part}'")
            seen[name] = part


def find_duplicate_files(split: Split) -> List[Tuple[str, str]]:
    """Arquivos com conteúdo idêntico (hash) em partições diferentes = vazamento exato."""
    first_seen: Dict[str, Tuple[str, str]] = {}
    dups: List[Tuple[str, str]] = []
    for part, ids in split.items():
        for files in ids.values():
            for f in files:
                h = hashlib.sha256(Path(f).read_bytes()).hexdigest()
                if h in first_seen and first_seen[h][0] != part:
                    dups.append((first_seen[h][1], f))
                else:
                    first_seen.setdefault(h, (part, f))
    return dups


def _group_by_label(labels: Sequence[int]) -> Dict[int, List[int]]:
    by_id: Dict[int, List[int]] = {}
    for i, l in enumerate(labels):
        by_id.setdefault(int(l), []).append(i)
    return by_id


def make_eval_pairs(labels: Sequence[int], n_pos: int = 3000, n_neg: int = 3000,
                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Pares (i, j) para verificação 1:1. y=1: mesma pessoa (fotos diferentes); y=0: pessoas diferentes."""
    rng = np.random.default_rng(seed)
    by_id = _group_by_label(labels)
    multi = [l for l, idx in by_id.items() if len(idx) >= 2]
    ids = list(by_id)
    if not multi or len(ids) < 2:
        raise ValueError("são necessárias >= 2 identidades e alguma com >= 2 imagens")
    pos, neg = [], []
    for _ in range(n_pos):
        l = multi[rng.integers(len(multi))]
        a, b = rng.choice(by_id[l], size=2, replace=False)
        pos.append((int(a), int(b)))
    for _ in range(n_neg):
        i, j = rng.choice(len(ids), size=2, replace=False)
        neg.append((int(rng.choice(by_id[ids[i]])), int(rng.choice(by_id[ids[j]]))))
    pairs = np.array(pos + neg, dtype=np.int64)
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return pairs, y


def open_set_protocol(labels: Sequence[int], enroll_k: int = 3, known_fraction: float = 0.5,
                      seed: int = 0) -> dict:
    """Protocolo de identificação 1:N com 'desconhecidos' (spec 004/006).

    - Identidades com >= enroll_k+1 fotos podem ser "conhecidas": `enroll_k` fotos viram o cadastro
      (protótipo) e as demais viram probes conhecidos.
    - Todas as outras identidades são "desconhecidas": todas as suas fotos são probes desconhecidos.
    """
    rng = np.random.default_rng(seed)
    by_id = _group_by_label(labels)
    eligible = sorted(l for l, idx in by_id.items() if len(idx) >= enroll_k + 1)
    rng.shuffle(eligible)
    n_known = max(1, int(len(eligible) * known_fraction))
    known = eligible[:n_known]
    unknown = [l for l in by_id if l not in set(known)]
    if not known or not unknown:
        raise ValueError("protocolo aberto precisa de identidades conhecidas e desconhecidas")

    enroll_idx: Dict[int, List[int]] = {}
    probe_idx: List[int] = []
    probe_known: List[bool] = []
    for l in known:
        idx = [int(i) for i in rng.permutation(by_id[l])]
        enroll_idx[l] = idx[:enroll_k]
        for i in idx[enroll_k:]:
            probe_idx.append(i)
            probe_known.append(True)
    for l in unknown:
        for i in by_id[l]:
            probe_idx.append(i)
            probe_known.append(False)
    labels_arr = np.asarray(labels)
    return {
        "known_ids": known,
        "unknown_ids": unknown,
        "enroll_idx": enroll_idx,
        "probe_idx": np.array(probe_idx, dtype=np.int64),
        "probe_known": np.array(probe_known, dtype=bool),
        "probe_labels": labels_arr[np.array(probe_idx, dtype=np.int64)],
    }
