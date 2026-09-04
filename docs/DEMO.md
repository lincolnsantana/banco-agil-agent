# Roteiro de demonstração — Banco Ágil (5–8 minutos)

Dados fictícios: CPF `11144477735`, nascimento `20/05/1990`. Sem chave de LLM.

## 0:00–0:45 — Abertura e arquitetura

- Apresente a ideia: quatro especialistas internos (Triagem, Crédito, Entrevista,
  Câmbio) sob uma única conversa.
- Mostre o fluxo em uma frase: `UI -> ConversationService -> Graph -> Tools ->
  Services -> CSV/SQLite/HTTP`.
- Destaque: apresentação e três especialistas com Groq (fallback determinístico),
  triagem sem LLM, no máximo 1 chamada por turno especialista, tools com estado
  injetado e CPF sempre vindo do estado confiável.

## 0:45–2:00 — Triagem + Crédito (consulta e aumento)

1. `streamlit run app.py`.
2. Mostre a apresentação inicial (já pede o CPF) e as ações alinhadas acima da entrada.
3. Envie `11144477735` direto na abertura → localiza o CPF e pede o nascimento.
5. Envie `20/05/1990` → `Dados confirmados. Como posso ajudar hoje?`
6. Envie `qual é meu limite?` → `Seu limite atual é R$ 2.500,00...`
7. Envie `quero aumentar meu limite` → pergunta naturalmente o limite total.
8. Envie `4000` → pedido aprovado e registrado (limite cadastrado inalterado).

Comente: decisão por faixa de score em `score_limite.csv`, uma linha por pedido.

## 2:00–4:00 — Rejeição, entrevista e reanálise

1. Peça `15000` → rejeitado para o score atual, com oferta de entrevista.
2. Responda `sim` e complete: `20000`, `formal`, `1000`, `0`, `não`.
3. Na última resposta, o score sobe para 1000 e a reanálise aprova no mesmo turno.

Comente: consentimento obrigatório, uma pergunta por vez, fórmula determinística,
sem promessa de aprovação.

## 4:00–5:00 — Câmbio e encerramento

1. Envie `cotação do dólar` → par, valor, fonte e horário (com rede).
2. Sem rede: a mesma pergunta retorna indisponibilidade controlada, sem valor
   inventado.
3. Envie `encerrar` → atendimento finalizado; `Reiniciar atendimento` limpa a
   conversa sem apagar os CSVs.

## 5:00–6:30 — Bastidores (código e testes)

- `src/banco_agil/agents/`: nós finos + `router.py`/`graph.py` (LangGraph).
- `src/banco_agil/tools/banking.py`: sete tools, dependências ocultas do LLM.
- `src/banco_agil/prompts/`: system único global + especialista, estado sem PII.
- Rode `pytest --cov=src/banco_agil --cov-report=term-missing` (≈ 219 testes) e
  `ruff format --check . && ruff check . && mypy src`.
- Mostre `docs/TEST_PLAN.md` (CT01–CT06) e `tests/e2e/test_journeys.py`.

## 6:30–8:00 — Fechamento

- Recapitule: autenticação obrigatória, reanálise automática, câmbio honesto,
  auditoria sem PII em SQLite, tudo testável offline.
- Perguntas.
