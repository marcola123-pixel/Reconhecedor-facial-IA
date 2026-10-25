"""Verificação automática dos itens TÉCNICOS do checklist de privacidade (Anexo B do enunciado).

    python scripts/check_privacy.py            # na raiz do repositório; retorna código 1 se achar problema

Confere: (1) .gitignore protege os diretórios privados; (2) nada proibido está versionado AGORA;
(3) nada proibido já passou pelo HISTÓRICO do Git (arquivo apagado depois continua no histórico!);
(4) nenhum segredo literal (token/senha) em arquivos de texto versionados.

Isso NÃO substitui as verificações humanas: consentimento assinado, guarda dos termos, licenças dos
datasets, autorizações do vídeo e envio de dados a serviços externos.
"""
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REQUIRED_IGNORE = ["data/private/", "embeddings/", "consent/", "*.pt", "*.npz", "classes.json",
                   "certs/", ".env", "logs/", "runs/"]
FORBIDDEN = [
    (r"\.(pt|pth|ckpt|onnx|safetensors|npz|db|sqlite|pem|key)$", "modelo/embedding/banco/chave"),
    (r"(^|/)(data/(private|raw|processed)|dataset|dataset_bruto|embeddings|consent|faces|database|logs|runs|certs)/",
     "pasta de dados privados"),
    (r"(^|/)classes\.json$", "lista de nomes das pessoas"),
    (r"\.(jpg|jpeg|bmp)$", "fotografia"),
    (r"(^|/)\.env$", "arquivo de segredos"),
    (r"(termo|consentimento).*\.(pdf|docx?|png|jpe?g)$", "termo de consentimento"),
]
PNG_OK = ("docs/", "experiments/results/figures/")
TEXT_SUFFIX = {".py", ".js", ".md", ".yaml", ".yml", ".txt", ".json", ".toml", ".cfg", ".html", ".sh"}
SECRET_RE = re.compile(r"(token|secret|api[_-]?key|passw(or)?d)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]", re.I)


def _git(repo: Path, *args: str) -> List[str]:
    out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def _bad(path: str):
    for pattern, why in FORBIDDEN:
        if re.search(pattern, path, re.I):
            return why
    if path.lower().endswith(".png") and not path.startswith(PNG_OK):
        return "imagem fora de docs/ e de experiments/results/figures/"
    return None


def scan(repo) -> Dict[str, List[str]]:
    repo = Path(repo)
    res: Dict[str, List[str]] = {"gitignore": [], "versionados": [], "historico": [], "segredos": []}
    gi = repo / ".gitignore"
    text = gi.read_text() if gi.exists() else ""
    for pat in REQUIRED_IGNORE:
        if pat not in text:
            res["gitignore"].append(f".gitignore não contém '{pat}'")
    tracked = _git(repo, "ls-files")
    for p in tracked:
        why = _bad(p)
        if why:
            res["versionados"].append(f"{p}  ({why})")
        f = repo / p
        if f.suffix.lower() in TEXT_SUFFIX and f.is_file() and f.stat().st_size < 2_000_000:
            for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                if SECRET_RE.search(line):
                    res["segredos"].append(f"{p}:{i}")
    try:
        hist = set(_git(repo, "log", "--all", "--name-only", "--pretty=format:"))
    except subprocess.CalledProcessError:
        hist = set()                                         # repositório sem commits
    for p in sorted(hist):
        why = _bad(p)
        if why and p not in tracked:
            res["historico"].append(f"{p}  ({why}) — já passou pelo histórico; exige reescrever o histórico")
    return res


def main() -> int:
    repo = Path.cwd()
    try:
        _git(repo, "rev-parse", "--git-dir")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Execute na raiz de um repositório Git (git init) com git instalado.")
        return 2
    res = scan(repo)
    problems = sum(len(v) for v in res.values())
    titles = {"gitignore": ".gitignore", "versionados": "Arquivos proibidos versionados agora",
              "historico": "Arquivos proibidos no HISTÓRICO", "segredos": "Possíveis segredos literais"}
    for k, items in res.items():
        print(f"[{'OK' if not items else 'PROBLEMA'}] {titles[k]}")
        for it in items:
            print("    -", it)
    print("\nNenhum problema técnico encontrado." if not problems else f"\n{problems} problema(s) encontrado(s).")
    print("Lembrete: consentimentos, licenças, vídeo e serviços externos são verificações HUMANAS.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
