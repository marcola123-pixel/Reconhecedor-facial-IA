"""Avalia um modelo treinado no dataset PRIVADO do grupo, imprimindo/salvando SÓ métricas agregadas.

    python scripts/evaluate_private.py --run runs/e04_cnn_triplet --root data/private/processed

Nenhum nome, caminho ou imagem sai deste script (exigência de LGPD do enunciado). Usa os limiares
escolhidos na validação PÚBLICA (meta.json), nunca limiares ajustados nos dados privados.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from facerec.data import FaceDataset  # noqa: E402
from facerec.evaluation import embed_dataset, identification_eval, pair_scores  # noqa: E402
from facerec.inference import load_run  # noqa: E402
from facerec.metrics import verification_report  # noqa: E402
from facerec.splits import scan_identities  # noqa: E402
from facerec.utils import to_py  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default="data/private/processed")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-pairs", type=int, default=2000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta, tf = load_run(args.run, device)
    ids = scan_identities(args.root, min_images=2)
    ds = FaceDataset(ids, tf)                      # rótulos viram índices anônimos
    emb = embed_dataset(model, ds, device)
    pos, neg = pair_scores(emb, ds.labels, args.n_pairs, 0)
    rep = verification_report(pos, neg, fixed_thresholds=meta["thresholds"])
    ident = identification_eval(emb, ds.labels,
                                {"enroll_k": 3, "known_fraction": 0.5, "protocol_seed": 0},
                                threshold=meta["id_threshold"])
    out = {"experimento": meta["name"], "identidades": len(ids), "imagens": len(ds),
           "verificacao": rep, "identificacao": ident,
           "obs": "limiares vindos da validação do dataset público"}
    text = json.dumps(to_py(out), indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
