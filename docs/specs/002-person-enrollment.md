# 002 — Cadastro de pessoa (sem retreinar)

**Status:** implementada · **Código:** `src/facerec/gallery.py`, `scripts/enroll.py`, `app/main.py`

## Problema
No desenho inicial (classificador softmax) cada pessoa nova exigia retreinar a rede. Queremos cadastrar
uma pessoa com poucas fotos, em segundos.

## Requisitos
- R1. Cadastrar uma pessoa a partir de N fotos: detectar → embedding → protótipo (média normalizada).
- R2. Cadastrar mais fotos da mesma pessoa reforça o protótipo.
- R3. **Remover** uma pessoa apaga todo vestígio dela da galeria e do arquivo salvo (LGPD, retirada de consentimento).
- R4. Armazenar apenas o necessário (minimização): protótipo e contagem, nunca as fotos (ADR-002).

## Critérios de aceitação
- [ ] Pessoa nova é reconhecida sem alterar pesos do modelo (`test_cadastra_e_identifica_sem_retreino`).
- [ ] Após `remove`, a pessoa não é identificada e o nome não existe no arquivo (`test_remover_pessoa_apaga_tudo`).
- [ ] A galeria salva (`embeddings/gallery.npz`) está no `.gitignore`.
