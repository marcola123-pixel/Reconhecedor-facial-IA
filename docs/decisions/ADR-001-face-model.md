# ADR-001 — Modelo de rosto e função de perda

**Status:** aceita (revisar após E01–E07)

## Contexto
O enunciado pede um pipeline detecção → embedding → comparação e quer que o grupo **investigue** em vez de apenas usar um modelo pronto.

## Decisão
- Detecção: **MTCNN** (facenet-pytorch).
- Embedding: duas famílias comparadas — **CNN própria** (`SmallCNN`, 128-d, do zero) e **InceptionResnetV1** pré-treinada (512-d) com 3 níveis de ajuste (sem / parcial / completo).
- Losses comparadas: Cross-Entropy (baseline), **Contrastive** (margem 1,0), **Triplet** semi-hard (α=0,2) e **InfoNCE supervisionada** (τ=0,1). Embeddings L2-normalizados, comparação por cosseno.

## Alternativas
ArcFace/MobileFaceNet: boas candidatas, deixadas para trabalho futuro por exigirem mais dados/tempo.

## Consequências
- O pré-processamento é recorte com margem + redimensionamento; **alinhamento por landmarks não foi implementado**.
- Modelos pré-treinados foram treinados em outros datasets faciais: pode haver identidades em comum com o conjunto público de teste. Isso deve ser verificado e **declarado** na análise; verificar também a licença/termos dos pesos usados.
