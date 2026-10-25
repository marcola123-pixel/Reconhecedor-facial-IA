# 004 — Pessoa desconhecida

**Status:** implementada · **Código:** `gallery.py` (`identify`), `metrics.py` (`open_set_identification`)

## Problema
Um sistema de portaria não pode obrigar todo rosto a ser alguém cadastrado. Um classificador softmax
sempre escolhe uma das N classes, mesmo com alta confiança para um estranho.

## Requisitos
- R1. Se a maior similaridade < limiar, o resultado é **desconhecido** (acesso negado).
- R2. O limiar é escolhido **na validação** (ADR-003), nunca no teste nem no uso.
- R3. Avaliar em protocolo aberto: probes conhecidos (devem ser aceitos e identificados) e desconhecidos (devem ser rejeitados).

## Critérios de aceitação
- [ ] Similaridade abaixo do limiar ⇒ `Match.name is None` (`test_desconhecido_abaixo_do_limiar`).
- [ ] Métricas reportadas: DIR, FRR de conhecidos, falsa identificação e FAR de desconhecidos (`test_identificacao_aberta`).
- [ ] Probes de conhecidos nunca incluem as fotos usadas no cadastro (`test_protocolo_aberto_nao_mistura_cadastro_e_probe`).
