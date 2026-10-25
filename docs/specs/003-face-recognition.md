# 003 — Reconhecimento 1:N por embeddings

**Status:** implementada · **Código:** `models.py`, `losses.py`, `gallery.py`, `evaluation.py`

## Requisitos
- R1. Uma rede converte um rosto em um embedding L2-normalizado; mesma pessoa → próximos, pessoas diferentes → distantes.
- R2. Identificação 1:N: comparar o embedding com todos os protótipos da galeria por similaridade de cosseno
  e escolher o mais próximo.
- R3. A rede é treinada com aprendizado de métrica (contrastive, triplet, InfoNCE) e comparada a Cross-Entropy
  e a um modelo pré-treinado (spec 006).

## Critérios de aceitação
- [ ] `forward` devolve vetores de norma 1 (`test_smallcnn_embedding_normalizado`).
- [ ] Losses se comportam como a teoria (`tests/test_losses.py`).
- [ ] `Gallery.identify` devolve o nome e a similaridade do protótipo mais próximo.
