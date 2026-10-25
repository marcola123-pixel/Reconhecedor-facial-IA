"""Métricas de reconhecimento facial (somente numpy). Convenção: aceita se score >= limiar.

FAR = fração de pares de PESSOAS DIFERENTES aceitos (falsa aceitação)
FRR = fração de pares da MESMA PESSOA rejeitados (falsa rejeição)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def far_frr_curve(pos, neg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.sort(np.asarray(pos, dtype=float))
    neg = np.sort(np.asarray(neg, dtype=float))
    u = np.unique(np.concatenate([pos, neg]))
    thr = np.concatenate([[u[0] - 1e-6], u, [u[-1] + 1e-6]])
    far = 1.0 - np.searchsorted(neg, thr, side="left") / len(neg)   # P(neg >= t)
    frr = np.searchsorted(pos, thr, side="left") / len(pos)         # P(pos <  t)
    return thr, far, frr


def equal_error_rate(pos, neg) -> Tuple[float, float]:
    """Retorna (EER, limiar no EER)."""
    thr, far, frr = far_frr_curve(pos, neg)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2.0), float(thr[i])


def roc_auc(pos, neg) -> float:
    """P(score_pos > score_neg), contando empates como 1/2 (equivale à AUC da ROC)."""
    pos = np.asarray(pos, dtype=float)
    neg = np.sort(np.asarray(neg, dtype=float))
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(neg)))


def threshold_at_far(neg, target_far: float) -> float:
    """Menor limiar cuja FAR (nos negativos) é <= target_far."""
    neg_s = np.sort(np.asarray(neg, dtype=float))
    cand = np.concatenate([np.unique(neg_s), [neg_s[-1] + 1e-6]])
    far = 1.0 - np.searchsorted(neg_s, cand, side="left") / len(neg_s)
    return float(cand[np.where(far <= target_far)[0][0]])


def rates_at_threshold(pos, neg, thr: float) -> Dict[str, float]:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    tp = int((pos >= thr).sum()); fn = len(pos) - tp
    fp = int((neg >= thr).sum()); tn = len(neg) - fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "threshold": float(thr), "far": fp / len(neg), "frr": fn / len(pos),
        "accuracy": (tp + tn) / (len(pos) + len(neg)),
        "precision": prec, "recall": rec, "f1": f1,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
    }


def verification_report(pos, neg, far_targets=(0.01, 0.001),
                        fixed_thresholds: Optional[Dict[str, float]] = None) -> dict:
    """AUC + EER e, para cada limiar, taxas. Se `fixed_thresholds` for dado (ex.: escolhidos na
    VALIDAÇÃO), usa esses limiares em vez de recalcular — é assim que se reporta o TESTE."""
    eer, thr_eer = equal_error_rate(pos, neg)
    rep = {"auc": roc_auc(pos, neg), "eer": eer, "eer_threshold": thr_eer,
           "mean_sim_same": float(np.mean(pos)), "mean_sim_diff": float(np.mean(neg))}
    if fixed_thresholds is None:
        fixed_thresholds = {"eer": thr_eer}
        for t in far_targets:
            fixed_thresholds[f"far{t:g}"] = threshold_at_far(neg, t)
    rep["at_threshold"] = {k: rates_at_threshold(pos, neg, v) for k, v in fixed_thresholds.items()}
    return rep


def _rate(mask: np.ndarray) -> Optional[float]:
    return float(mask.mean()) if mask.size else None


def open_set_identification(probe_emb, probe_labels, probe_known, proto_matrix, proto_labels,
                            threshold: float) -> dict:
    """Identificação 1:N com rejeição. Embeddings já L2-normalizados -> produto interno = cosseno.

    dir          : conhecido, aceito E identificado corretamente (taxa de detecção+identificação)
    frr_known    : conhecido rejeitado como desconhecido
    misid_known  : conhecido aceito, porém identificado como OUTRA pessoa
    far_unknown  : desconhecido aceito (deixaria um estranho entrar)
    """
    sims = np.asarray(probe_emb) @ np.asarray(proto_matrix).T
    best = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)
    pred = np.asarray(proto_labels)[best]
    accepted = best_sim >= threshold
    known = np.asarray(probe_known, dtype=bool)
    probe_labels = np.asarray(probe_labels)
    right = pred == probe_labels
    return {
        "threshold": float(threshold),
        "n_known_probes": int(known.sum()), "n_unknown_probes": int((~known).sum()),
        "dir": _rate((accepted & right)[known]),
        "frr_known": _rate((~accepted)[known]),
        "misid_known": _rate((accepted & ~right)[known]),
        "far_unknown": _rate(accepted[~known]),
        "top1_closed_set": _rate(right[known]),
    }


def choose_identification_threshold(probe_emb, probe_labels, probe_known, proto_matrix,
                                    proto_labels, n_candidates: int = 200) -> float:
    """Limiar de identificação escolhido na VALIDAÇÃO: onde frr_known ≈ far_unknown."""
    sims = np.asarray(probe_emb) @ np.asarray(proto_matrix).T
    lo, hi = float(sims.max(axis=1).min()), float(sims.max(axis=1).max())
    best_t, best_gap = lo, np.inf
    for t in np.linspace(lo, hi, n_candidates):
        r = open_set_identification(probe_emb, probe_labels, probe_known, proto_matrix, proto_labels, t)
        gap = abs((r["frr_known"] or 0.0) - (r["far_unknown"] or 0.0))
        if gap < best_gap:
            best_t, best_gap = float(t), gap
    return best_t
