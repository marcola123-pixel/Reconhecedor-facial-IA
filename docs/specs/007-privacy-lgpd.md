# 007 — Privacidade e LGPD (critério eliminatório)

**Status:** vigente em todo o projeto

## Requisitos
- R1. Só se usam fotos de pessoas **adultas que consentiram por escrito**; os termos ficam **fora do Git**.
- R2. É proibido publicar: fotos, nomes associados, embeddings, galeria, checkpoints derivados de dados privados, termos de consentimento, logs com pessoas.
- R3. Os modelos são treinados **apenas com dataset público**; dados privados servem para teste/demonstração (checkpoints ficam locais mesmo assim).
- R4. Dados privados são processados **localmente**: nenhuma foto, embedding ou checkpoint é enviado a serviços externos (inclusive assistentes de IA, que recebem só código, specs e dados sintéticos).
- R5. Resultados privados somente agregados, sem identificar participantes.
- R6. Deve ser possível excluir os dados de uma pessoa (`scripts/enroll.py --remove`, `DELETE /pessoas/{nome}`).
- R7. Ao fim da disciplina: apagar fotos, embeddings, galeria e checkpoints.

## Critérios de aceitação
- [ ] `.gitignore` cobre `data/private/`, `data/raw/`, `data/processed/`, `embeddings/`, `consent/`, `dataset/`, `dataset_bruto/`, `classes.json`, `*.pt`, `*.npz`, `certs/`.
- [ ] `git log --all -- data embeddings consent '*.pt'` não retorna nada (verificar o **histórico**, não só a árvore atual).
- [ ] README informa fonte, finalidade, licença e uso de cada dataset público.
