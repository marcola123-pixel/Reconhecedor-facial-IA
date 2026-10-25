"""Spec 003/004/006: métricas corretas em casos cujo resultado é conhecido de antemão."""
import numpy as np

from facerec.metrics import (choose_identification_threshold, equal_error_rate, far_frr_curve,
                             open_set_identification, rates_at_threshold, roc_auc, threshold_at_far,
                             verification_report)


def test_separacao_perfeita():
    pos, neg = np.array([0.8, 0.9, 0.95]), np.array([0.1, 0.2, 0.3])
    assert roc_auc(pos, neg) == 1.0
    eer, thr = equal_error_rate(pos, neg)
    assert eer == 0.0 and 0.3 < thr <= 0.8


def test_distribuicoes_iguais_auc_meio():
    x = np.linspace(0, 1, 1000)
    assert abs(roc_auc(x, x) - 0.5) < 1e-9
    eer, _ = equal_error_rate(x, x)
    assert abs(eer - 0.5) < 0.01


def test_auc_inversa_e_empates():
    assert roc_auc([0.1, 0.2], [0.8, 0.9]) == 0.0
    assert roc_auc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_far_frr_monotonicas():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(0.7, 0.1, 500), rng.normal(0.3, 0.1, 500)
    thr, far, frr = far_frr_curve(pos, neg)
    assert np.all(np.diff(far) <= 1e-12) and np.all(np.diff(frr) >= -1e-12)
    assert far[0] == 1.0 and frr[0] == 0.0 and far[-1] == 0.0 and frr[-1] == 1.0


def test_threshold_at_far_respeita_alvo():
    rng = np.random.default_rng(1)
    neg = rng.normal(0.3, 0.1, 10000)
    t = threshold_at_far(neg, 0.01)
    assert (neg >= t).mean() <= 0.01
    assert (neg >= t).mean() > 0.005      # não é conservador demais


def test_taxas_e_matriz_de_confusao():
    pos, neg = np.array([0.9, 0.8, 0.4, 0.7]), np.array([0.1, 0.6, 0.2, 0.3])
    r = rates_at_threshold(pos, neg, 0.5)
    assert r["confusion"] == {"tp": 3, "fn": 1, "fp": 1, "tn": 3}
    assert r["far"] == 0.25 and r["frr"] == 0.25 and r["accuracy"] == 0.75
    assert abs(r["f1"] - 0.75) < 1e-12


def test_limiar_fixo_vindo_da_validacao_e_usado_no_teste():
    rng = np.random.default_rng(2)
    val_pos, val_neg = rng.normal(0.7, 0.1, 500), rng.normal(0.3, 0.1, 500)
    te_pos, te_neg = rng.normal(0.7, 0.1, 500), rng.normal(0.3, 0.1, 500)
    val = verification_report(val_pos, val_neg)
    fixed = {k: v["threshold"] for k, v in val["at_threshold"].items()}
    test = verification_report(te_pos, te_neg, fixed_thresholds=fixed)
    for k in fixed:                                        # o teste NÃO recalcula limiares
        assert test["at_threshold"][k]["threshold"] == fixed[k]


def _toy_gallery():
    protos = np.array([[1.0, 0.0], [0.0, 1.0]])           # pessoas 0 e 1
    probe = np.array([[0.99, 0.14], [0.14, 0.99], [0.7, 0.7], [-1.0, 0.0]])
    probe = probe / np.linalg.norm(probe, axis=1, keepdims=True)
    labels = np.array([0, 1, 5, 6])                         # 5 e 6 são desconhecidos
    known = np.array([True, True, False, False])
    return probe, labels, known, protos, np.array([0, 1])


def test_identificacao_aberta():
    probe, labels, known, protos, plabels = _toy_gallery()
    r = open_set_identification(probe, labels, known, protos, plabels, threshold=0.9)
    assert r["dir"] == 1.0 and r["frr_known"] == 0.0 and r["misid_known"] == 0.0
    assert r["far_unknown"] == 0.0                          # desconhecidos (cos ~0.7 e -1) rejeitados
    r2 = open_set_identification(probe, labels, known, protos, plabels, threshold=0.5)
    assert r2["far_unknown"] == 0.5                         # o de cos 0.7 passa a ser aceito


def test_escolha_de_limiar_de_identificacao_separa_conhecidos_de_desconhecidos():
    probe, labels, known, protos, plabels = _toy_gallery()
    t = choose_identification_threshold(probe, labels, known, protos, plabels)
    r = open_set_identification(probe, labels, known, protos, plabels, t)
    assert r["frr_known"] == 0.0 and r["far_unknown"] == 0.0
