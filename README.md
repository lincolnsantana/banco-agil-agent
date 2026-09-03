# Banco Ágil — Chat bancário com especialistas internos

## Visão Geral

O Banco Ágil é um banco digital fictício cujo atendimento ao cliente é feito por
quatro especialistas internos de IA, apresentados ao cliente como uma única
conversa contínua em Streamlit:

- **Triagem**: recepciona, autentica (CPF + nascimento em `clientes.csv`) e
  direciona; encerra após a terceira falha sem revelar o campo incorreto.
- **Crédito**: consulta o limite atual e processa pedidos de aumento
  (aprova/rejeita pela faixa de score; registra em
  `solicitacoes_aumento_limite.csv` sem alterar o limite cadastrado).
- **Entrevista de Crédito**: com consentimento, coleta 5 dados financeiros, um
  por vez, recalcula o score por fórmula determinística, atualiza
  `clientes.csv` e devolve ao Crédito para reanálise.
- **Câmbio**: consulta a cotação atual via API REST/JSON (AwesomeAPI); nunca
  inventa valor nem recomenda investimento.

Tudo roda offline nos testes (mocks + fixtures temporárias). Nenhuma credencial
real é necessária; sem chave do provedor, os fluxos determinísticos funcionam e
intenções ambíguas recebem pedido de esclarecimento.

## Arquitetura, agentes, fluxos e manipulação de dados

```text
UI (app.py) -> ConversationService -> Graph (LangGraph) -> Tools -> Services
Services -> protocolos de repository/integration -> CSV / SQLite / HTTP
```

- **UI** (`app.py`): sessão e histórico entre reruns, máscara de CPF/nascimento
  na exibição, erros recuperáveis genéricos, botões Encerrar/Reiniciar. Sem
  regra de negócio.
- **Grafo** (`agents/router.py`, `agents/graph.py`): entrada exige autenticação;
  triagem continua no mesmo turno para o especialista; entrevista concluída
  retorna ao crédito para reanálise; `MAX_HANDLER_STEPS=2` + `recursion_limit=8`
  impedem loops; histórico limitado às 6 mensagens recentes.
- **Nós** (`agents/triage.py`, `credit.py`, `credit_interview.py`,
  `exchange.py`): parsers determinísticos primeiro (CPF, data ISO, valores
  `R$`, sim/não, pares de moedas); no máximo **uma** chamada LLM por turno,
  só para intenção livre inconclusiva, com saída estruturada.
- **Prompts** (`prompts/`): um system message = global + especialista ativo +
  estado mínimo sanitizado (sem PII, < 500 caracteres); só as tools do
  especialista ativo são expostas; IDs/versões testados contra `PROMPTS.md`.
- **Tools** (`tools/banking.py`): `authenticate_client`, `get_credit_limit`,
  `request_limit_increase`, `update_credit_score`, `get_exchange_rate`,
  `end_service`. Estado e serviços são injetados (`InjectedToolArg`) e ficam
  fora do schema visível ao LLM; cada tool protegida revalida autenticação e o
  CPF vem sempre do estado confiável.
- **Dados**: `clientes.csv`, `score_limite.csv` (faixas 0–1000 sem lacunas),
  `solicitacoes_aumento_limite.csv` (uma linha por pedido:
  `pendente` → `aprovado`/`rejeitado`); escrita sob `filelock` + arquivo
  temporário + `os.replace`; `Decimal` e timestamps UTC ISO 8601.
- **Auditoria** (`repositories/audit_sqlite.py`, `observability/`): eventos de
  turno (início, transição, erro, fim) em SQLite local e métricas de LLM
  (modelo, versão de prompt, latência, tokens); nunca contém PII; falha de
  auditoria nunca quebra o atendimento.

## Funcionalidades implementadas

- Autenticação com 3 tentativas e encerramento cordial.
- Consulta de limite e solicitação de aumento com decisão por score.
- Entrevista de crédito completa com reanálise automática.
- Cotação de moedas com tratamento de indisponibilidade.
- Encerramento (`encerrar`, `sair`, `finalizar`…) prioritário em qualquer nó.
- UI Streamlit com sessão persistente, Reiniciar (limpa a conversa sem apagar
  persistência) e mascaramento de dados sensíveis.
- Auditoria técnica consultável por sessão + métricas que distinguem turnos com
  0 e 1 chamada LLM.
- 219 testes (unitários, integração e E2E) + `docs/TEST_PLAN.md` de homologação.

## Desafios enfrentados e soluções

- **Pydantic validava dependências injetadas** e rejeitava fakes estruturais:
  `SkipValidation` nos serviços injetados, mantendo tipos concretos por escopo.
- **Invariante `ended`/`end_reason`**: atribuições separadas quebravam a
  validação — transição atômica em `ConversationState.end()`.
- **Texto livre com PII chegando ao LLM**: sanitização por lista positiva de
  termos + no máximo 5 mensagens anteriores sanitizadas.
- **Reanálise pós-entrevista sem duplicar regra**: sinal transitório
  `credit_reanalysis_pending` + `update_credit_score` via tool.
- **Falsos positivos no encerramento** (`quero sair das dívidas`): encerramento
  por frases completas, não por substring.
- **Warnings de `sqlite3.Connection` não fechada**: `contextlib.closing` no
  repositório e nos testes (suíte limpa até com `-W error`).
- **`data/clientes.csv` ausente**: criado com dados fictícios para a jornada.

## Escolhas técnicas e justificativas

| Área | Escolha | Motivo |
| --- | --- | --- |
| Orquestração | LangGraph | Grafo explícito de rotas + limite de recursão |
| LLM | LangChain Core + Groq atrás de adaptador | Troca de provedor sem tocar nos nós; testes offline |
| Validação | Pydantic + pydantic-settings | Contratos tipados em domínio, estado, CSV e `.env` |
| Câmbio | HTTPX + AwesomeAPI configurável | Cliente REST tipado com retry e erros controlados |
| Persistência | CSV (negócio) + SQLite (auditoria) | Exigência do desafio + diagnóstico sem PII |
| Concorrência | filelock + `os.replace` | Escrita atômica em arquivos mutáveis |
| Qualidade | Ruff + Mypy estrito + Pytest + RESPX | Contrato de cada tarefa do projeto |

**Economia de tokens**: rota determinística, parsers e templates antes do LLM;
máximo 1 chamada/turno; system message único (global + especialista);
somente tools do especialista ativo; estado sem PII (≤ 500 caracteres);
temperatura `0.1`, saída de 180 tokens, timeout de 20 s; métricas por chamada.

**Limitações**: sem RAG/banco vetorial (fora do escopo); sem checkpoint de
sessão persistente (T019, opcional); câmbio exige rede; LLM é exceção para
intenção ambígua, nunca decide regra de negócio.

## Tutorial de execução e testes

Pré-requisitos: Python 3.12+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # opcional; sem GROQ_API_KEY os fluxos seguem determinísticos
```

Interface (dados fictícios: CPF `11144477735`, nascimento `1990-05-20`):

```bash
streamlit run app.py
```

Roteiro na UI: informe o CPF → informe o nascimento → `qual é meu limite?`
(`R$ 2.500,00`) → `quero aumentar meu limite` → `4000` → `Encerrar atendimento`
ou `Reiniciar atendimento`. Demonstração completa em `docs/DEMO.md`;
homologação em `docs/TEST_PLAN.md`.

Validação:

```bash
ruff format --check .
ruff check .
mypy src
pytest --cov=src/banco_agil --cov-report=term-missing
pytest -q tests/integration tests/e2e
```

Docker/CI (opcional): `docker compose config`, `docker compose build`,
`docker compose up --build`; o workflow `.github/workflows/ci.yml` repete a
validação sem chaves e sem serviços externos.
