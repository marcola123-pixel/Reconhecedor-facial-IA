"""Spec 002/003/004: cadastro sem retreino, desconhecido por limiar, remoção real (LGPD)."""
import numpy as np

from facerec.gallery import Gallery


def _emb(rng, center, n=5, noise=0.05):
    return center + rng.normal(0, noise, size=(n, len(center)))


def test_cadastra_e_identifica_sem_retreino():
    rng = np.random.default_rng(0)
    ana, bia = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])
    g = Gallery()
    g.enroll("ana", _emb(rng, ana))
    g.enroll("bia", _emb(rng, bia))
    assert g.names() == ["ana", "bia"]
    m = g.identify(_emb(rng, ana, n=1)[0], threshold=0.8)
    assert m.name == "ana" and m.score > 0.9
    g.enroll("caio", _emb(rng, np.array([0, 0, 1.0, 0])))       # pessoa nova: nada é retreinado
    assert g.identify(np.array([0, 0, 1.0, 0.02]), 0.8).name == "caio"


def test_desconhecido_abaixo_do_limiar():
    g = Gallery()
    g.enroll("ana", np.array([[1.0, 0.0, 0.0]]))
    m = g.identify(np.array([0.0, 1.0, 0.0]), threshold=0.5)
    assert m.name is None and m.best_name == "ana" and m.score < 0.5


def test_galeria_vazia():
    assert Gallery().identify(np.array([1.0, 0.0]), 0.5).name is None


def test_remover_pessoa_apaga_tudo(tmp_path):
    g = Gallery()
    g.enroll("ana", np.array([[1.0, 0.0]]))
    g.enroll("bia", np.array([[0.0, 1.0]]))
    assert g.remove("ana") is True and g.remove("ana") is False
    assert g.identify(np.array([1.0, 0.0]), 0.5).name is None
    path = tmp_path / "gal.npz"
    g.save(path)
    assert Gallery.load(path).names() == ["bia"]               # nada de 'ana' no arquivo
    assert b"ana" not in path.read_bytes()


def test_salvar_e_carregar_preserva_prototipos(tmp_path):
    g = Gallery()
    g.enroll("ana", np.array([[1.0, 0.1], [0.9, 0.0]]))
    g.save(tmp_path / "g.npz")
    h = Gallery.load(tmp_path / "g.npz")
    a, b = g.prototypes()[1], h.prototypes()[1]
    assert np.allclose(a, b)
    assert Gallery.load(tmp_path / "nao_existe.npz").names() == []
