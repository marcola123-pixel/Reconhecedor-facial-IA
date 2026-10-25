"""Lê experiments/results/*.json e gera a tabela de resultados do README (sem inventar números).

    python scripts/aggregate_results.py            # imprime a tabela
    python scripts/aggregate_results.py --write    # atualiza o bloco entre os marcadores do README
"""
import argparse
import json
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def row(r: dict) -> str:
    cfg, t, it = r["config"], r["test"], r["identification"]["test"]
    eer_rates = t["at_threshold"]["eer"]
    if cfg["model"]["type"] == "facenet":
        model = "InceptionResnetV1 (pré-treinada)"
        strat = {"frozen": "sem fine-tuning", "partial": "fine-tuning parcial",
                 "full": "fine-tuning completo"}[cfg["model"]["finetune"]]
    else:
        model = f"CNN própria ({cfg['model']['embedding_dim']}-d)"
        strat = "treino do zero"
    loss = cfg["loss"]["type"]
    return (f"| {r['name'].split('_')[0].upper()} | {model} | {strat} + {loss} | público | "
            f"{fmt(t['auc'])} | {fmt(t['eer'])} | {fmt(eer_rates['accuracy'])} | {fmt(eer_rates['f1'])} | "
            f"{fmt(it['dir'])} / {fmt(it['far_unknown'])} | {r['training']['best_epoch']} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.results).glob("e*.json"))
    head = ("| Exp. | Modelo | Estratégia | Dataset | AUC | EER | Accuracy | F1 | DIR / FAR-desc. | Melhor época |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    table = head + "\n" + "\n".join(row(json.loads(p.read_text())) for p in files) if files else \
        head + "\n| — | nenhum experimento executado ainda | | | | | | | | |"
    print(table)
    if args.write:
        rd = Path(args.readme)
        text = rd.read_text()
        a, b = text.index(START) + len(START), text.index(END)
        rd.write_text(text[:a] + "\n" + table + "\n" + text[b:])
        print(f"\nREADME atualizado ({len(files)} experimento(s)).")


if __name__ == "__main__":
    main()
