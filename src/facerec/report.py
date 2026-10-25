"""Gráficos agregados (distribuição de similaridade e FAR/FRR). Não contêm dados individuais."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import far_frr_curve


def save_figures(name: str, pos, neg, out_dir: str | Path, thresholds: dict | None = None) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(min(np.min(pos), np.min(neg)), max(np.max(pos), np.max(neg)), 50)
    ax.hist(neg, bins=bins, alpha=0.6, label="pessoas diferentes")
    ax.hist(pos, bins=bins, alpha=0.6, label="mesma pessoa")
    for k, t in (thresholds or {}).items():
        ax.axvline(t, ls="--", lw=1, color="k")
        ax.text(t, ax.get_ylim()[1] * 0.95, k, rotation=90, va="top", ha="right", fontsize=7)
    ax.set_xlabel("similaridade de cosseno"); ax.set_ylabel("pares"); ax.set_title(f"{name} — teste")
    ax.legend(); fig.tight_layout()
    p = out / f"{name}_similaridade.png"; fig.savefig(p, dpi=130); plt.close(fig); paths.append(str(p))

    thr, far, frr = far_frr_curve(pos, neg)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(thr, far, label="FAR"); ax.plot(thr, frr, label="FRR")
    ax.set_xlabel("limiar"); ax.set_ylabel("taxa"); ax.set_title(f"{name} — FAR × FRR (teste)")
    ax.legend(); fig.tight_layout()
    p = out / f"{name}_far_frr.png"; fig.savefig(p, dpi=130); plt.close(fig); paths.append(str(p))
    return paths
