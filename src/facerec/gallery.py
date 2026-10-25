"""Galeria de pessoas cadastradas (spec 002 e ADR-002).

Minimização de dados: guardamos APENAS o protótipo (soma dos embeddings + contagem) de cada pessoa,
nunca as fotos. Remover uma pessoa apaga tudo o que a representa (direito de retirada da LGPD).
O arquivo salvo é dado biométrico PRIVADO: fica no .gitignore (`embeddings/`, `*.npz`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Match:
    name: Optional[str]        # None => desconhecido
    score: float               # cosseno com o protótipo mais próximo
    best_name: Optional[str]   # protótipo mais próximo (mesmo se rejeitado) — use só para depuração


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class Gallery:
    def __init__(self) -> None:
        self._sum: Dict[str, np.ndarray] = {}
        self._count: Dict[str, int] = {}

    def names(self) -> List[str]:
        return sorted(self._sum)

    def __len__(self) -> int:
        return len(self._sum)

    def enroll(self, name: str, embeddings: np.ndarray) -> None:
        """Cadastra (ou reforça) uma pessoa a partir de embeddings de algumas fotos."""
        e = _l2(np.atleast_2d(np.asarray(embeddings, dtype=np.float64)))
        self._sum[name] = self._sum.get(name, 0.0) + e.sum(axis=0)
        self._count[name] = self._count.get(name, 0) + len(e)

    def remove(self, name: str) -> bool:
        existed = name in self._sum
        self._sum.pop(name, None)
        self._count.pop(name, None)
        return existed

    def prototypes(self):
        names = self.names()
        mat = _l2(np.stack([self._sum[n] for n in names])) if names else np.zeros((0, 0))
        return names, mat

    def identify(self, embedding: np.ndarray, threshold: float) -> Match:
        names, mat = self.prototypes()
        if not names:
            return Match(None, -1.0, None)
        sims = mat @ _l2(np.asarray(embedding, dtype=np.float64))
        i = int(np.argmax(sims))
        score = float(sims[i])
        return Match(names[i] if score >= threshold else None, score, names[i])

    def save(self, path: str | Path) -> None:
        names = self.names()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, names=np.array(names, dtype=str),
                 sums=np.stack([self._sum[n] for n in names]) if names else np.zeros((0, 0)),
                 counts=np.array([self._count[n] for n in names], dtype=np.int64))

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        g = cls()
        p = Path(path)
        if not p.exists():
            return g
        with np.load(p, allow_pickle=False) as z:
            for n, s, c in zip(z["names"], z["sums"], z["counts"]):
                g._sum[str(n)] = s.astype(np.float64)
                g._count[str(n)] = int(c)
        return g
