# ADR-002 — Armazenamento dos embeddings

**Status:** aceita

## Contexto
Embeddings faciais são dado biométrico privado (enunciado, seção 20). A LGPD exige minimização e possibilidade de exclusão.

## Decisão
Guardar por pessoa **apenas** a soma dos embeddings e a contagem (o protótipo é a soma normalizada), em `embeddings/gallery.npz`
(NumPy, `allow_pickle=False`), fora do Git. Nenhuma foto é persistida pelo cadastro nem pelo app.

## Alternativas
Guardar todos os embeddings (permitiria k-NN) — rejeitado: mais dados biométricos do que o necessário.
Banco SQL/vetorial — desnecessário para dezenas de pessoas; reavaliar se a escala crescer.

## Consequências
Remover uma pessoa = apagar uma entrada. Não é possível "desfazer" uma foto específica do protótipo.
