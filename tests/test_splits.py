"""Spec 006: separação por identidade, sem leakage; pares e protocolo aberto bem formados."""
import numpy as np

from facerec.splits import (assert_disjoint_identities, find_duplicate_files, make_eval_pairs,
                            open_set_protocol, scan_identities, split_identities)


def _fake_dataset(root, n_ids=20, n_imgs=5):
    for i in range(n_ids):
        d = root / f"pessoa_{i:02d}"
        d.mkdir(parents=True)
        for j in range(n_imgs):
            (d / f"{j}.png").write_bytes(f"id{i}-img{j}".encode())   # conteúdo único por arquivo
    return root


def test_scan_respeita_min_images(tmp_path):
    _fake_dataset(tmp_path, n_ids=3, n_imgs=2)
    (tmp_path / "pessoa_00" / "extra.png").write_bytes(b"x")
    assert len(scan_identities(tmp_path, min_images=2)) == 3
    assert list(scan_identities(tmp_path, min_images=3)) == ["pessoa_00"]


def test_split_e_disjunto_por_identidade(tmp_path):
    ids = scan_identities(_fake_dataset(tmp_path), min_images=2)
    split = split_identities(ids, (0.6, 0.2, 0.2), seed=1)
    assert_disjoint_identities(split)
    assert sum(len(v) for v in split.values()) == len(ids)
    assert all(len(v) >= 2 for v in split.values())


def test_split_e_reprodutivel(tmp_path):
    ids = scan_identities(_fake_dataset(tmp_path), min_images=2)
    a = split_identities(ids, seed=7)
    b = split_identities(ids, seed=7)
    c = split_identities(ids, seed=8)
    assert {k: sorted(v) for k, v in a.items()} == {k: sorted(v) for k, v in b.items()}
    assert {k: sorted(v) for k, v in a.items()} != {k: sorted(v) for k, v in c.items()}


def test_detecta_arquivo_duplicado_entre_particoes(tmp_path):
    ids = scan_identities(_fake_dataset(tmp_path), min_images=2)
    split = split_identities(ids, seed=3)
    assert find_duplicate_files(split) == []
    # copia o conteúdo de uma imagem do treino para uma identidade do teste => vazamento
    f_train = next(iter(next(iter(split["train"].values()))))
    f_test = next(iter(next(iter(split["test"].values()))))
    open(f_test, "wb").write(open(f_train, "rb").read())
    assert len(find_duplicate_files(split)) == 1


def test_pares_de_verificacao():
    labels = [0, 0, 0, 1, 1, 2, 2, 2, 3]
    pairs, y = make_eval_pairs(labels, n_pos=50, n_neg=50, seed=0)
    lab = np.array(labels)
    for (a, b), yy in zip(pairs, y):
        assert (lab[a] == lab[b]) == bool(yy)
        if yy == 1:
            assert a != b           # mesma pessoa, mas fotos diferentes
    assert y.sum() == 50 and len(y) == 100


def test_protocolo_aberto_nao_mistura_cadastro_e_probe():
    labels = np.repeat(np.arange(10), [5, 5, 5, 5, 5, 5, 2, 2, 2, 1])
    p = open_set_protocol(labels, enroll_k=3, known_fraction=0.5, seed=0)
    enrolled = {i for idx in p["enroll_idx"].values() for i in idx}
    assert enrolled.isdisjoint(set(p["probe_idx"].tolist()))          # probe != foto de cadastro
    assert all(len(v) == 3 for v in p["enroll_idx"].values())
    known, unknown = set(p["known_ids"]), set(p["unknown_ids"])
    assert known.isdisjoint(unknown)
    # probes desconhecidos só vêm de identidades desconhecidas
    assert set(p["probe_labels"][~p["probe_known"]].tolist()) <= unknown
    assert set(p["probe_labels"][p["probe_known"]].tolist()) <= known
