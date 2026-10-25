# ADR-004 — Separação dos dados por identidade

**Status:** aceita

## Contexto
No primeiro protótipo, treino/validação eram sorteados por **imagem**: fotos da mesma pessoa e da mesma sessão ficavam nos dois lados (data leakage),
inflando a acurácia e não medindo o que importa (reconhecer alguém **novo**).

## Decisão
Dividir por **identidade** (70/15/15, seed fixa). Pessoas do teste nunca aparecem no treino. Verificar identidades disjuntas e arquivos duplicados por hash antes de treinar.

## Consequências
Não detecta quase-duplicatas (mesma foto recomprimida). Mitigação: usar datasets curados e revisar amostras manualmente.
