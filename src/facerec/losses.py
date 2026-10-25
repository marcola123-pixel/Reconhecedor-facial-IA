"""Funções de perda de aprendizado de métrica (spec 006 / ADR-001).

Todas recebem embeddings z já L2-normalizados (distâncias euclidianas em [0, 2]) e rótulos de
identidade. Cross-Entropy fica no treino (precisa da cabeça classificadora).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_dist(z: torch.Tensor) -> torch.Tensor:
    """Distância euclidiana (B,B) com gradiente estável (evita NaN em d=0 na diagonal)."""
    sq = (z * z).sum(dim=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (z @ z.T)
    return torch.sqrt(d2.clamp(min=0.0) + 1e-12)


def contrastive_loss(z1, z2, y, margin: float = 1.0):
    """L = y·d² + (1−y)·max(0, m−d)².  y=1: mesma pessoa (aproxima); y=0: diferentes (afasta até o raio m)."""
    d = F.pairwise_distance(z1, z2, eps=1e-8)
    y = y.float()
    return (y * d.pow(2) + (1.0 - y) * F.relu(margin - d).pow(2)).mean()


def contrastive_loss_batch(z, labels, margin: float = 1.0, balance: bool = True):
    """Contrastive sobre TODOS os pares i<j do batch. `balance` pesa positivos e negativos igualmente
    (num batch P×K os negativos são ~95% dos pares)."""
    d = pairwise_dist(z)
    same = labels[:, None] == labels[None, :]
    idx = torch.arange(z.size(0), device=z.device)
    upper = idx[:, None] < idx[None, :]          # pares i<j
    l_pos = d[same & upper].pow(2)
    l_neg = F.relu(margin - d[~same & upper]).pow(2)
    if balance:
        terms = [t.mean() for t in (l_pos, l_neg) if t.numel() > 0]
        return sum(terms) / len(terms) if terms else z.sum() * 0.0
    allv = torch.cat([l_pos, l_neg])
    return allv.mean() if allv.numel() else z.sum() * 0.0


def triplet_loss(a, p, n, margin: float = 0.2):
    """L = max(0, d(a,p) − d(a,n) + α). Tripla (âncora, positivo, negativo) já montada."""
    return F.relu(F.pairwise_distance(a, p, eps=1e-8) - F.pairwise_distance(a, n, eps=1e-8) + margin).mean()


def triplet_loss_semihard(z, labels, margin: float = 0.2):
    """Triplet com mineração semi-hard dentro do batch (FaceNet).

    Para cada par (âncora a, positivo p) escolhe o negativo n mais próximo que ainda seja mais
    distante que p (d(a,p) < d(a,n)); se não existir, usa o negativo mais distante. Triplas "fáceis"
    (loss 0) não ensinam nada — por isso a mineração.
    """
    B = z.size(0)
    d = pairwise_dist(z)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = same & ~eye                                   # (a,p)
    neg_mask = ~same                                         # (a,n)

    d_ap = d.unsqueeze(2)                                    # (B,B,1)
    d_an = d.unsqueeze(1).expand(B, B, B)                    # (B,B,B) -> [a,p,n] = d(a,n)
    valid = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)    # tripla [a,p,n] válida
    semihard = valid & (d_an > d_ap)

    big = torch.finfo(d.dtype).max
    d_semi = torch.where(semihard, d_an, torch.full_like(d_an, big)).min(dim=2).values
    d_far = torch.where(valid, d_an, torch.full_like(d_an, -big)).max(dim=2).values
    d_neg = torch.where(semihard.any(dim=2), d_semi, d_far)  # (B,B)

    has_triplet = valid.any(dim=2)                           # (a,p) com ao menos 1 negativo
    if not has_triplet.any():
        return z.sum() * 0.0
    losses = F.relu(d - d_neg + margin)[has_triplet]
    return losses.mean()


def supcon_loss(z, labels, tau: float = 0.1):
    """InfoNCE supervisionada (SupCon): para cada âncora, o(s) positivo(s) competem contra todos os
    outros exemplos do batch (negativos) num softmax com temperatura τ. s = cosseno (z normalizado).
    L_i = −média_p∈P(i) log[ exp(s_ip/τ) / Σ_{k≠i} exp(s_ik/τ) ]"""
    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = (z @ z.T) / tau
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    n_pos = pos.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return z.sum() * 0.0
    log_prob = log_prob.masked_fill(~pos, 0.0)
    return (-(log_prob.sum(dim=1)[valid] / n_pos[valid])).mean()


def build_metric_loss(cfg: dict):
    """Retorna f(z, labels) -> loss para contrastive | triplet | supcon."""
    kind = cfg["type"]
    if kind == "contrastive":
        return lambda z, y: contrastive_loss_batch(z, y, cfg.get("margin", 1.0), cfg.get("balance", True))
    if kind == "triplet":
        return lambda z, y: triplet_loss_semihard(z, y, cfg.get("margin", 0.2))
    if kind == "supcon":
        return lambda z, y: supcon_loss(z, y, cfg.get("tau", 0.1))
    raise ValueError(f"loss desconhecida: {kind}")
