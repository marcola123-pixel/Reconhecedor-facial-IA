# 005 — Interface de câmera (Parte 2)

**Status:** em adaptação — revisar quando o enunciado da Parte 2 estiver completo · **Código:** `app/`

## Requisitos
- R1. Página web que abre a câmera do navegador, envia frames ao servidor e mostra o resultado.
- R2. A **decisão** liberado/negado é tomada no **servidor** (limiar do `meta.json`), não no navegador.
- R3. HTTPS, cabeçalhos de segurança (CSP, HSTS…), CORS restrito, servidor só em `127.0.0.1`.
- R4. Endpoints administrativos (cadastrar/remover pessoa) exigem `PORTEIRO_TOKEN`; sem token configurado ficam desativados.

## Critérios de aceitação
- [ ] `POST /reconhecer` devolve `liberado`, `nome` (só se liberado) e `similaridade`.
- [ ] `DELETE /pessoas/{nome}` remove a pessoa da galeria persistida.
- [ ] Nenhuma foto é gravada em disco pelo servidor.
