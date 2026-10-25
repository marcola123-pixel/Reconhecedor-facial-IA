"""Spec 006: as losses de métrica se comportam como a teoria prevê (precisa de PyTorch)."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from facerec.losses import (contrastive_loss, contrastive_loss_batch, supcon_loss,  # noqa: E402
                            triplet_loss, triplet_loss_semihard)


def test_contrastive_pares():
    a = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    same = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    far = torch.tensor([[3.0, 0.0], [3.0, 0.0]])        # d = 3 > margem 1
    near = torch.tensor([[0.5, 0.0], [0.5, 0.0]])       # d = 0.5 < margem 1
    y1, y0 = torch.ones(2), torch.zeros(2)
    assert contrastive_loss(a, same, y1).item() < 1e-6      # positivo coincidente: sem custo
    assert contrastive_loss(a, far, y0).item() == 0.0       # negativo fora do raio: sem custo
    assert abs(contrastive_loss(a, far, y1).item() - 9.0) < 1e-4        # positivo longe: d² = 9
    assert abs(contrastive_loss(a, near, y0).item() - 0.25) < 1e-4      # negativo dentro do raio: (1-0.5)²


def test_triplet_exemplo_da_aula():
    a, p, n = torch.tensor([[0.0]]), torch.tensor([[2.0]]), torch.tensor([[5.0]])
    assert triplet_loss(a, p, n, margin=0.0).item() == 0.0          # max(0, 2-5) = 0
    assert abs(triplet_loss(a, n, p, margin=0.0).item() - 3.0) < 1e-4   # tripla violada: 5-2 = 3


def _two_clusters(sep=1.0, noise=0.01, k=4):
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([[1.0, 0.0], [-1.0, 0.0]]) * sep
    z = torch.cat([c + noise * torch.randn(k, 2, generator=g) for c in centers])
    return F.normalize(z, dim=1), torch.tensor([0] * k + [1] * k)


def test_semihard_zero_quando_bem_separado_e_positivo_quando_misturado():
    z, y = _two_clusters()
    assert triplet_loss_semihard(z, y, margin=0.2).item() == 0.0
    # classes intercaladas no círculo: positivos a 90° e negativos a 45° => triplas violadas
    ang = torch.deg2rad(torch.tensor([0., 90., 180., 270., 45., 135., 225., 315.]))
    zm = torch.stack([ang.cos(), ang.sin()], dim=1)
    assert triplet_loss_semihard(zm, y, margin=0.2).item() > 0.0


def test_margem_evita_colapso():
    z = torch.ones(8, 2) / (2 ** 0.5)                   # todos os embeddings iguais
    y = torch.tensor([0] * 4 + [1] * 4)
    assert abs(triplet_loss_semihard(z, y, margin=0.2).item() - 0.2) < 5e-3   # loss = margem, não 0


def test_supcon_menor_em_clusters_do_que_em_ruido():
    zc, y = _two_clusters()
    zr = F.normalize(torch.randn(8, 2, generator=torch.Generator().manual_seed(2)), dim=1)
    assert supcon_loss(zc, y, tau=0.1).item() < supcon_loss(zr, y, tau=0.1).item()


@pytest.mark.parametrize("fn", [
    lambda z, y: triplet_loss_semihard(z, y, 0.2),
    lambda z, y: supcon_loss(z, y, 0.1),
    lambda z, y: contrastive_loss_batch(z, y, 1.0),
])
def test_gradientes_finitos(fn):
    z = F.normalize(torch.randn(16, 8, generator=torch.Generator().manual_seed(3)), dim=1).requires_grad_(True)
    y = torch.arange(4).repeat_interleave(4)
    loss = fn(z, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


def test_batch_sem_positivos_nao_quebra():
    z = F.normalize(torch.randn(4, 3), dim=1)
    y = torch.arange(4)                                  # nenhuma pessoa repetida
    assert supcon_loss(z, y).item() == 0.0
    assert triplet_loss_semihard(z, y).item() == 0.0
