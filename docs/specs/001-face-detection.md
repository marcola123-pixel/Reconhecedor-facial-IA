# 001 — Detecção e pré-processamento de rosto

**Status:** implementada · **Código:** `src/facerec/detection.py`, `scripts/prepare_faces.py`

## Problema
Fotos e frames de câmera contêm fundo, vários rostos e tamanhos variados. A rede de embeddings precisa
receber sempre o mesmo formato: um rosto recortado.

## Requisitos
- R1. Detectar rosto em uma imagem RGB e devolver o recorte (com margem) redimensionado.
- R2. Se houver vários rostos, usar o maior; se nenhum, sinalizar "sem rosto" (sem lançar erro).
- R3. O pré-processamento do treino, da avaliação e da aplicação deve ser **idêntico** (mesmo recorte,
  mesma normalização: `build_transform`).

## Critérios de aceitação
- [ ] `FaceDetector.crop()` retorna `PIL.Image` ou `None`.
- [ ] `prepare_faces.py` informa quantas imagens falharam e não interrompe o lote.
- [ ] Treino, `evaluate_private.py` e o app usam a mesma função `build_transform` (`tests/test_models_data.py`).

## Fora de escopo
Alinhamento por landmarks (rotação pelos olhos) — ver ADR-001, "consequências".
