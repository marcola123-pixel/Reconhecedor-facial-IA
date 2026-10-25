# ADR-003 — Limiar de reconhecimento

**Status:** aceita

## Decisão
- Dois limiares, ambos escolhidos **somente na validação**:
  1. verificação 1:1 — ponto de EER e pontos de FAR-alvo (1% e 0,1%);
  2. identificação 1:N — onde FRR dos conhecidos ≈ FAR dos desconhecidos (protocolo aberto).
- Os limiares são gravados em `runs/<exp>/meta.json` e **fixados** no teste, no dataset privado e no app.
- Para portaria, o ponto de operação deve priorizar **FAR baixo** (não deixar estranho entrar), aceitando FRR maior.

## Consequências
A taxa de falsa aceitação do mundo real depende da diferença entre a distribuição do dataset público e da câmera real; por isso o
dataset privado entra como teste final agregado.
