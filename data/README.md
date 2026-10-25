# Dados

Nenhum dado é versionado neste repositório. Esta pasta só descreve **de onde obter** os dados
e **onde colocá-los localmente** (todas as pastas abaixo estão no `.gitignore`).

```
data/
├── raw/<dataset>/<identidade>/*.jpg        # dataset público, baixado da fonte oficial
├── processed/<dataset>/<identidade>/*.png  # rostos detectados e recortados (gerado por scripts/prepare_faces.py)
└── private/                                # dataset do grupo (consentimento!) — NUNCA publicar
    ├── raw/<id>/*.jpg
    └── processed/<id>/*.png
```

## Dataset público

> Preencha esta tabela com os termos **lidos na fonte oficial** antes de usar (exigência do enunciado).

| Item | LFW — Labeled Faces in the Wild |
|---|---|
| Fonte oficial | https://vis-www.cs.umass.edu/lfw/ |
| Finalidade neste trabalho | treino, validação e teste dos modelos (benchmark) |
| Licença / termos de uso | **Uso acadêmico e de pesquisa permitidos pela UMASS; livre de custos para fins não comerciais, sendo vedada a redistribuição direta de cópias modificadas sem autorização.** com o texto da fonte oficial |
| Redistribuição | **não** incluímos cópias no repositório; cada pessoa baixa da fonte |
| Como obter | `torchvision.datasets.LFWPeople(root="data/raw", download=True)` ou download manual |

Alternativa maior: **CelebA** (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) — termos próprios
(uso acadêmico/não comercial, redistribuição restrita). Se usar, converta as imagens para
`<identidade>/*.jpg` usando `identity_CelebA.txt`.

Depois de baixar:

```bash
python scripts/prepare_faces.py --src data/raw/lfw_funneled --dst data/processed/lfw
```

## Dataset privado (grupo)

- Fotos tiradas **pelos próprios alunos**, com **consentimento assinado** e participantes **maiores de 18 anos**.
- Termos de consentimento ficam **fora do Git** (guardados em local privado e seguro).
- Colete o mínimo necessário (minimização) e varie luz, pose, distância, fundo e momento.
- Resultados com esses dados aparecem **somente agregados** (`scripts/evaluate_private.py`).
- Ao final da disciplina: apagar fotos, embeddings, galeria e checkpoints derivados.
