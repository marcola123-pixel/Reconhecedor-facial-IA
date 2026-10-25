"""Treino + avaliação de UM experimento, dirigido por um arquivo YAML (spec 006).

    python -m facerec.train configs/e04_cnn_triplet.yaml

Fluxo: split por identidade -> checagem de leakage -> (treino) -> escolha da melhor época pela
VALIDAÇÃO -> limiares escolhidos na VALIDAÇÃO -> métricas finais no TESTE com esses limiares.
O conjunto de teste nunca participa de treino, seleção de modelo nem escolha de limiar.
"""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import FaceDataset, PKBatchSampler, build_transform
from .evaluation import embed_dataset, identification_eval, pair_scores
from .losses import build_metric_loss
from .metrics import equal_error_rate, roc_auc, verification_report
from .models import ClassifierHead, build_model, count_parameters
from .report import save_figures
from .splits import (assert_disjoint_identities, find_duplicate_files, scan_identities,
                     split_identities)
from .utils import environment_info, set_seed, to_py

DEFAULTS = {
    "name": "exp",
    "hypothesis": "",
    "seed": 42,
    "device": None,
    "data": {"root": "data/processed/lfw", "min_images": 4, "fractions": [0.7, 0.15, 0.15],
             "split_seed": 42, "image_size": 64},
    "model": {"type": "small_cnn", "embedding_dim": 128, "pretrained": "vggface2", "finetune": "frozen"},
    "loss": {"type": "triplet", "margin": 0.2, "tau": 0.1, "balance": True},
    "train": {"epochs": 30, "P": 16, "K": 4, "steps_per_epoch": 100, "optimizer": "adam",
              "lr": 1e-3, "weight_decay": 1e-4, "num_workers": 0},
    "eval": {"n_pairs": 3000, "pair_seed": 0, "enroll_k": 3, "known_fraction": 0.5,
             "protocol_seed": 0, "far_targets": [0.01, 0.001], "batch_size": 128},
    "output": {"runs_dir": "runs", "results_dir": "experiments/results"},
}


def deep_update(base: dict, new: dict) -> dict:
    out = deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    return deep_update(DEFAULTS, yaml.safe_load(Path(path).read_text()))


def _val_metrics(model, ds_val, device, ecfg):
    emb = embed_dataset(model, ds_val, device, ecfg["batch_size"])
    pos, neg = pair_scores(emb, ds_val.labels, ecfg["n_pairs"], ecfg["pair_seed"])
    eer, thr = equal_error_rate(pos, neg)
    return {"val_eer": eer, "val_auc": roc_auc(pos, neg), "val_eer_threshold": thr}


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    d, mcfg, lcfg, tcfg, ecfg = cfg["data"], cfg["model"], cfg["loss"], cfg["train"], cfg["eval"]
    name = cfg["name"]
    print(f"[{name}] dispositivo: {device}")

    # ---- dados: split por identidade + verificação de leakage -------------------------------
    identities = scan_identities(d["root"], d["min_images"])
    split = split_identities(identities, d["fractions"], d["split_seed"])
    assert_disjoint_identities(split)
    dups = find_duplicate_files(split)
    if dups:
        raise RuntimeError(f"data leakage: {len(dups)} arquivo(s) idêntico(s) entre partições")
    tf_train, tf_eval = build_transform(d["image_size"], True), build_transform(d["image_size"], False)
    ds_train = FaceDataset(split["train"], tf_train)
    ds_val = FaceDataset(split["val"], tf_eval)
    ds_test = FaceDataset(split["test"], tf_eval)
    data_stats = {p: {"identities": len(ids), "images": sum(len(v) for v in ids.values())}
                  for p, ids in split.items()}
    print(f"[{name}] dados: {data_stats}")

    # ---- modelo ------------------------------------------------------------------------------
    model = build_model(mcfg, d["image_size"]).to(device)
    head = None
    if lcfg["type"] == "ce":
        head = ClassifierHead(model.embedding_dim, len(ds_train.class_names)).to(device)
    metric_loss = build_metric_loss(lcfg) if lcfg["type"] in {"contrastive", "triplet", "supcon"} else None
    params = [p for p in model.parameters() if p.requires_grad] + (list(head.parameters()) if head else [])
    n_params = count_parameters(model)

    # ---- treino ------------------------------------------------------------------------------
    history = [{"epoch": 0, **_val_metrics(model, ds_val, device, ecfg)}]   # referência sem treino
    best = {"val_eer": float("inf"), "epoch": 0, "state": None}
    t0 = time.time()
    if tcfg["epochs"] > 0 and lcfg["type"] != "none" and params:
        opt_cls = torch.optim.Adam if tcfg["optimizer"] == "adam" else torch.optim.SGD
        kw = {"lr": tcfg["lr"], "weight_decay": tcfg["weight_decay"]}
        if tcfg["optimizer"] != "adam":
            kw["momentum"] = 0.9
        opt = opt_cls(params, **kw)
        sampler = PKBatchSampler(ds_train.labels, tcfg["P"], tcfg["K"], tcfg["steps_per_epoch"], cfg["seed"])
        loader = DataLoader(ds_train, batch_sampler=sampler, num_workers=tcfg["num_workers"])
        for epoch in range(1, tcfg["epochs"] + 1):
            model.train()
            if head is not None:
                head.train()
            losses = []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                if head is not None:                                    # E02: Cross-Entropy
                    loss = F.cross_entropy(head(model.features(x)), y)
                else:                                                   # losses de métrica
                    loss = metric_loss(model(x), y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            m = _val_metrics(model, ds_val, device, ecfg)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **m})
            print(f"[{name}] época {epoch:3d}  loss={np.mean(losses):.4f}  val_EER={m['val_eer']:.4f}  "
                  f"val_AUC={m['val_auc']:.4f}")
            if m["val_eer"] < best["val_eer"]:       # seleção SOMENTE pela validação
                best = {"val_eer": m["val_eer"], "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        model.load_state_dict(best["state"])
    train_seconds = time.time() - t0

    # ---- avaliação final: limiares da VALIDAÇÃO, métricas no TESTE ---------------------------
    emb_val = embed_dataset(model, ds_val, device, ecfg["batch_size"])
    emb_test = embed_dataset(model, ds_test, device, ecfg["batch_size"])
    pos_v, neg_v = pair_scores(emb_val, ds_val.labels, ecfg["n_pairs"], ecfg["pair_seed"])
    pos_t, neg_t = pair_scores(emb_test, ds_test.labels, ecfg["n_pairs"], ecfg["pair_seed"])
    val_rep = verification_report(pos_v, neg_v, tuple(ecfg["far_targets"]))
    thresholds = {k: v["threshold"] for k, v in val_rep["at_threshold"].items()}
    test_rep = verification_report(pos_t, neg_t, fixed_thresholds=thresholds)
    test_rep["eer_threshold_test_only"] = test_rep.pop("eer_threshold")     # não usar para decidir

    id_val = identification_eval(emb_val, ds_val.labels, ecfg)              # escolhe o limiar aqui
    id_test = identification_eval(emb_test, ds_test.labels, ecfg, threshold=id_val["threshold"])

    # ---- artefatos ---------------------------------------------------------------------------
    run_dir = Path(cfg["output"]["runs_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "best.pt")          # checkpoint: fica fora do Git (.gitignore)
    meta = {"name": name, "model": mcfg, "image_size": d["image_size"],
            "thresholds": thresholds, "id_threshold": id_val["threshold"]}
    (run_dir / "meta.json").write_text(json.dumps(to_py(meta), indent=2))

    results = {
        "name": name, "hypothesis": cfg["hypothesis"], "config": cfg, "environment": environment_info(),
        "parameters": n_params, "data": data_stats,
        "leakage_check": {"identities_disjoint": True, "duplicate_files": 0},
        "training": {"best_epoch": best["epoch"], "seconds": round(train_seconds, 1), "history": history},
        "validation": val_rep, "test": test_rep,
        "identification": {"val": id_val, "test": id_test},
    }
    res_dir = Path(cfg["output"]["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{name}.json").write_text(json.dumps(to_py(results), indent=2, ensure_ascii=False))
    save_figures(name, pos_t, neg_t, res_dir / "figures", {k: v for k, v in thresholds.items()})
    t = test_rep
    print(f"[{name}] TESTE  AUC={t['auc']:.4f}  EER={t['eer']:.4f}  "
          f"acc@EER={t['at_threshold']['eer']['accuracy']:.4f}  DIR={id_test['dir']}  "
          f"FAR_desc={id_test['far_unknown']}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--device", default=None, help="cpu | cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    run(cfg)


if __name__ == "__main__":
    main()
