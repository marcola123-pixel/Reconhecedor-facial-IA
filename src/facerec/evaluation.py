"""Avaliação: verificação 1:1 (pares) e identificação 1:N com rejeição de desconhecidos."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import choose_identification_threshold, open_set_identification
from .splits import make_eval_pairs, open_set_protocol


@torch.no_grad()
def embed_dataset(model, dataset, device, batch_size: int = 128, num_workers: int = 0) -> np.ndarray:
    """Embeddings L2-normalizados, na mesma ordem de `dataset.labels`."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = [model(x.to(device)).cpu().numpy() for x, _ in loader]
    return np.concatenate(out).astype(np.float64)


def pair_scores(emb: np.ndarray, labels, n_pairs: int, seed: int):
    """Similaridade de cosseno em pares 'mesma pessoa' (pos) e 'pessoas diferentes' (neg)."""
    pairs, y = make_eval_pairs(labels, n_pairs, n_pairs, seed)
    s = (emb[pairs[:, 0]] * emb[pairs[:, 1]]).sum(axis=1)
    return s[y == 1], s[y == 0]


def identification_eval(emb: np.ndarray, labels, eval_cfg: dict, threshold: float | None = None) -> dict:
    """Protocolo aberto: cadastra `enroll_k` fotos de metade das pessoas elegíveis e testa com
    as demais fotos delas (conhecidos) + todas as fotos das outras pessoas (desconhecidos).
    Se `threshold` for None, ele é ESCOLHIDO aqui (use só na validação)."""
    proto = open_set_protocol(labels, eval_cfg["enroll_k"], eval_cfg["known_fraction"], eval_cfg["protocol_seed"])
    ids = sorted(proto["enroll_idx"])
    mat = np.stack([emb[proto["enroll_idx"][l]].mean(axis=0) for l in ids])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    probes = emb[proto["probe_idx"]]
    if threshold is None:
        threshold = choose_identification_threshold(probes, proto["probe_labels"], proto["probe_known"],
                                                    mat, np.array(ids))
    res = open_set_identification(probes, proto["probe_labels"], proto["probe_known"], mat,
                                  np.array(ids), threshold)
    res["n_known_identities"] = len(proto["known_ids"])
    res["n_unknown_identities"] = len(proto["unknown_ids"])
    return res
