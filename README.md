# Reconhecedor Facial 1:N com aprendizado de representação

Trabalho 1 de Inteligência Artificial (UFMS). Pipeline: **imagem → detecção (MTCNN) → recorte → rede neural → embedding → comparação → reconhecido / desconhecido**.

Cada pessoa é uma região no espaço de embeddings; **cadastrar alguém novo não exige retreinar** (basta guardar o protótipo das fotos).
O que se investiga na Parte 1 é *qual rede e qual função de perda* produzem os embeddings que melhor separam pessoas **nunca vistas** no treino.

> **Estado atual:** código, specs, ADRs e testes prontos. **Nenhum experimento foi executado ainda** — a tabela de resultados abaixo é preenchida automaticamente por `scripts/aggregate_results.py` a partir dos JSONs em `experiments/results/`.

## Estrutura

```
configs/                 um YAML por experimento (hipótese + hiperparâmetros + seed)
src/facerec/             splits (por identidade), metrics, losses, models, data, train, evaluation, gallery, detection
scripts/                 prepare_faces, evaluate_private, enroll, aggregate_results, gerar_certificado
app/                     Parte 2: aplicação web (FastAPI + HTTPS)
tests/                   testes das losses, métricas, splits (anti-leakage), galeria
docs/specs/              especificações e critérios de aceitação (Spec-Driven Development)
docs/decisions/          ADRs (decisões e alternativas)
experiments/results/     resultados AGREGADOS (JSON + gráficos) — sem dados individuais
data/README.md           como obter os datasets (nenhum dado é versionado)
```

## Instalação

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt                      # se o pip reclamar de versões: pip install --no-deps facenet-pytorch
pip install -e .
pytest                                               # testes
```

## Dados

Veja [`data/README.md`](data/README.md): dataset **público** (LFW; CelebA opcional) baixado da fonte oficial, e dataset **privado** do grupo (consentimento, só adultos, nunca no Git).

## Parte 1 — como reproduzir os experimentos

```bash
# 1) baixar o LFW (fonte oficial) em data/raw/ e recortar os rostos
python scripts/prepare_faces.py --src data/raw/lfw_funneled --dst data/processed/lfw

# 2) rodar cada experimento (cada um grava experiments/results/<exp>.json + gráficos)
python -m facerec.train configs/e01_facenet_baseline.yaml
python -m facerec.train configs/e02_cnn_crossentropy.yaml
python -m facerec.train configs/e03_cnn_contrastive.yaml
python -m facerec.train configs/e04_cnn_triplet.yaml
python -m facerec.train configs/e05_cnn_supcon.yaml
python -m facerec.train configs/e06_facenet_partial.yaml   # ajuste a loss conforme E03–E05
python -m facerec.train configs/e07_facenet_full.yaml

# 3) atualizar a tabela abaixo
python scripts/aggregate_results.py --write

# 4) teste final, só agregado, no dataset privado (limiares vêm da validação pública)
python scripts/evaluate_private.py --run runs/<melhor_exp> --root data/private/processed
```

### Protocolo (spec 006)

- Divisão **por identidade** 70/15/15 (seed fixa): quem está no teste nunca esteve no treino. Checagem automática de identidades disjuntas e de arquivos idênticos (hash) entre partições.
- A **melhor época** e **todos os limiares** (EER, FAR 1%, FAR 0,1% e limiar de identificação) saem da **validação**; o teste só aplica esses valores.
- Métricas: AUC, EER, FAR/FRR, Accuracy, Precision, Recall, F1, matriz de confusão (pares), e — em protocolo aberto — DIR, FRR de conhecidos, falsa identificação e FAR de desconhecidos. Gráficos: distribuição de similaridade (mesma pessoa × pessoas diferentes) e FAR×FRR.
- Reprodutibilidade: seed, hiperparâmetros, versões das bibliotecas e hardware gravados em cada JSON.

### Resultados

<!-- RESULTS:START -->
| Exp. | Modelo | Estratégia | Dataset | AUC | EER | Accuracy | F1 | DIR / FAR-desc. | Melhor época |
|---|---|---|---|---|---|---|---|---|---|
| E01 | InceptionResnetV1 (pré-treinada) | sem fine-tuning + none | público | 0.989 | 0.022 | 0.979 | 0.979 | 0.950 / 0.062 | 0 |
| E02 | CNN própria (128-d) | treino do zero + ce | público | 0.834 | 0.243 | 0.758 | 0.763 | 0.086 / 0.464 | 27 |
| E03 | CNN própria (128-d) | treino do zero + contrastive | público | 0.910 | 0.167 | 0.837 | 0.840 | 0.275 / 0.348 | 30 |
| E04 | CNN própria (128-d) | treino do zero + triplet | público | 0.922 | 0.146 | 0.856 | 0.860 | 0.491 / 0.329 | 25 |
| E05 | CNN própria (128-d) | treino do zero + supcon | público | 0.935 | 0.138 | 0.865 | 0.866 | 0.514 / 0.348 | 28 |
| E06 | InceptionResnetV1 (pré-treinada) | fine-tuning parcial + triplet | público | 0.991 | 0.026 | 0.971 | 0.971 | 0.914 / 0.036 | 9 |
| E07 | InceptionResnetV1 (pré-treinada) | fine-tuning completo + triplet | público | 0.991 | 0.021 | 0.976 | 0.976 | 0.923 / 0.017 | 3 |

## 📊 Resumo dos Experimentos (Parte 1)

Abaixo estão consolidados os principais experimentos realizados no dataset público (LFW), avaliando diferentes arquiteturas, estratégias de fine-tuning e funções de perda:

| Exp. | Modelo | Estratégia / Loss | Dataset | Acurácia | F1-Score | Observação |
|:---:|---|---|:---:|:---:|:---:|---|
| **E01** | InceptionResnetV1 | Baseline (sem fine-tuning) | Público | 0.979 | 0.979 | Excelente ponto de partida sem custo de treino. |
| **E02** | CNN Própria (128-d) | Treino do zero + Cross-Entropy | Público | 0.758 | 0.763 | Dificuldade de generalização; modelo sofreu com poucas épocas. |
| **E03** | CNN Própria (128-d) | Treino do zero + Contrastive Loss | Público | 0.837 | 0.840 | Melhora considerável na separação de pares com métrica de distância. |
| **E04** | CNN Própria (128-d) | Treino do zero + Triplet Loss | Público | 0.856 | 0.860 | Superior à loss contrastiva; melhorou a estruturação do espaço vetorial. |
| **E05** | CNN Própria (128-d) | Treino do zero + SupCon Loss | Público | 0.865 | 0.866 | Melhor desempenho entre os modelos treinados do zero. |
| **E06** | InceptionResnetV1 | Fine-tuning parcial + Triplet | Público | 0.971 | 0.971 | Estável, mas congelar camadas limitou o aprendizado de características finas. |
| **E07** | InceptionResnetV1 | Fine-tuning completo + Triplet | Público | **0.976** | **0.976** | **Melhor configuração global (escolhida para a Parte 2)**; convergência na época 3. |

*Nota: Os resultados detalhados de AUC, EER e métricas por experimento encontram-se salvos na pasta `experiments/results/`.*
<!-- RESULTS:END -->

*Accuracy e F1 são de verificação de pares, no limiar de EER escolhido na validação. DIR/FAR-desc.: taxa de conhecidos aceitos e identificados corretamente / taxa de desconhecidos aceitos (protocolo aberto).*

### Análise

> Preencher após os experimentos: qual modelo venceu e por quê; efeito do fine-tuning; efeito do threshold; erros típicos; limitações; custo computacional × desempenho. Declarar também o risco de **sobreposição de identidades** entre os dados de pré-treino do modelo e o conjunto de teste (ADR-001).

## Parte 2 — aplicação

```bash
python scripts/gerar_certificado.py                                        # uma vez
python scripts/enroll.py --run runs/<melhor_exp> --name "<nome>" --images pasta_com_fotos/
export FACEREC_RUN=runs/<melhor_exp>
export PORTEIRO_TOKEN="<segredo seu>"      # habilita cadastro/remoção via API; nunca no código/Git
python app/main.py                         # https://localhost:8443
```

A decisão liberado/negado é do **servidor**; HTTPS, CSP/HSTS, CORS restrito, bind em `127.0.0.1`. Remover uma pessoa: `scripts/enroll.py --remove "<nome>"` ou `DELETE /pessoas/{nome}`.
O enunciado da Parte 2 ainda deve ser conferido contra a spec 005.

## Privacidade (LGPD) — critério eliminatório

Fotos, nomes, embeddings, galeria, checkpoints e termos de consentimento **não** vão para o Git (`.gitignore`). Modelos são treinados **só com dataset público**. Dados privados são processados localmente, aparecem apenas em métricas agregadas e são apagados ao fim da disciplina. Detalhes: [`docs/specs/007-privacy-lgpd.md`](docs/specs/007-privacy-lgpd.md).

## Spec-Driven Development e uso de IA

Requisitos → especificação → critérios de aceitação → implementação → testes, em `docs/specs/` e `docs/decisions/`. O código foi desenvolvido com assistência de IA (Claude) a partir dessas especificações; os integrantes são responsáveis por compreender e explicar arquitetura, losses, treino, métricas e decisões. Nenhuma foto, embedding ou checkpoint privado foi/deve ser enviado a assistentes de IA.
