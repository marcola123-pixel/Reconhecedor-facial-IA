# 006 — Protocolo experimental

**Status:** implementada · **Código:** `splits.py`, `train.py`, `evaluation.py`, `metrics.py`, `configs/`

## Pergunta
Qual configuração de modelo, treino e/ou fine-tuning dá o melhor reconhecimento facial para pessoas **novas**?

## Requisitos
- R1. Todo experimento é um YAML em `configs/` com **hipótese**, modelo, loss, hiperparâmetros e seed.
- R2. Separação **por identidade** em treino/validação/teste; conjuntos sem identidades em comum.
- R3. O teste **não** participa de treino, escolha da época, escolha de limiar nem escolha do modelo.
- R4. Checagem de leakage: identidades disjuntas e nenhum arquivo idêntico (hash) entre partições.
- R5. Métricas: AUC, EER, FAR, FRR, Accuracy, Precision, Recall, F1, matriz de confusão (pares), DIR/FAR de desconhecidos; histograma de similaridades.
- R6. Reprodutibilidade: seed, versões de bibliotecas e hardware gravados em `experiments/results/<exp>.json`.
- R7. Resultados com dados privados só agregados (`scripts/evaluate_private.py`).

## Experimentos planejados
E01 pré-treinado sem ajuste · E02 CNN + Cross-Entropy · E03 CNN + Contrastive · E04 CNN + Triplet semi-hard ·
E05 CNN + SupCon/InfoNCE · E06 pré-treinado + fine-tuning parcial · E07 pré-treinado + fine-tuning completo.

## Critérios de aceitação
- [ ] `split_identities` é reprodutível e disjunto (`tests/test_splits.py`).
- [ ] Limiares do teste vêm da validação (`test_limiar_fixo_vindo_da_validacao_e_usado_no_teste`).
- [ ] Cada JSON de resultado contém hipótese, config, ambiente e `leakage_check`.
