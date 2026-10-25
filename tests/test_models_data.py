"""Spec 001/006: modelo gera embeddings normalizados; amostrador P×K e dataset corretos."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from facerec.data import FaceDataset, PKBatchSampler, build_transform  # noqa: E402
from facerec.models import SmallCNN  # noqa: E402


def test_smallcnn_embedding_normalizado():
    m = SmallCNN(embedding_dim=32, image_size=64).eval()
    z = m(torch.randn(5, 3, 64, 64))
    assert z.shape == (5, 32)
    assert torch.allclose(z.norm(dim=1), torch.ones(5), atol=1e-5)


def test_pk_sampler_composicao():
    labels = np.repeat(np.arange(10), 6)                 # 10 pessoas x 6 fotos
    s = PKBatchSampler(labels, P=4, K=3, steps=5, seed=0)
    batches = list(s)
    assert len(batches) == 5
    for b in batches:
        assert len(b) == 12
        ids = labels[b]
        assert len(set(ids.tolist())) == 4                           # 4 pessoas distintas
        assert all((ids == i).sum() == 3 for i in set(ids.tolist()))  # 3 fotos de cada


def test_pk_sampler_pessoa_com_poucas_fotos_usa_reposicao():
    labels = np.array([0, 0, 1, 1, 1, 1])
    batch = next(iter(PKBatchSampler(labels, P=2, K=3, steps=1, seed=0)))
    assert len(batch) == 6


def test_dataset_e_transform(tmp_path):
    ids = {}
    for name in ("ana", "bia"):
        (tmp_path / name).mkdir()
        ids[name] = []
        for j in range(2):
            p = tmp_path / name / f"{j}.png"
            Image.fromarray(np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)).save(p)
            ids[name].append(str(p))
    ds = FaceDataset(ids, build_transform(64, train=False))
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and -1.0 <= x.min() and x.max() <= 1.0
    assert ds.labels.tolist() == [0, 0, 1, 1]
