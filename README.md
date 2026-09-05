# Banco Ágil — Chat bancário com especialistas internos

## Visão Geral

O Banco Ágil é um banco digital fictício cujo atendimento ao cliente é feito por
quatro especialistas internos de IA, apresentados ao cliente como uma única
conversa contínua em Streamlit:

- **Triagem**: explica a validação, verifica imediatamente o CPF no cadastro e
  pede novamente quando ele é inválido. Para CPF localizado, solicita nascimento
  em `DD/MM/AAAA` e só então autentica; encerra após a terceira falha.
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

- **UI** (`app.py`): tela inicial com campo central e quatro atalhos, transição
  animada para o chat, conversa que abre pela mensagem do cliente e responde com
  as boas-vindas e o pedido de CPF, sessão e histórico
  entre reruns, máscara de CPF/nascimento na exibição, erros recuperáveis
  genéricos e ações Encerrar/Reiniciar alinhadas. Sem regra de negócio.
- **Grafo** (`agents/router.py`, `agents/graph.py`): entrada exige autenticação;
  triagem continua no mesmo turno para o especialista; entrevista concluída
  retorna ao crédito para reanálise; `MAX_HANDLER_STEPS=2` + `recursion_limit=8`
  impedem loops; histórico limitado às 6 mensagens recentes.
- **Nós** (`agents/triage.py`, `credit.py`, `credit_interview.py`,
  `exchange.py`): autenticação e rotas claras são determinísticas (parser
  primeiro; `alterar/mudar/ajustar/modificar limite` é aumento, `score` /
  `entrevista` é entrevista); intenção pós-autenticação ambígua usa o Groq
  para classificar; Crédito, Entrevista e Câmbio podem ter a resposta final
  redigida pelo Groq, com fallback canônico. Em qualquer especialista, texto
  que não é resposta ao passo atual passa por um parser de recusa e de pedido
  novo e, se preciso, pelo Groq: "não quero mais aumento, quero o dólar" leva
  o cliente ao câmbio no mesmo turno em vez de repetir "qual limite?".
- **Prompts** (`prompts/`): um system message = global + especialista ativo +
  estado mínimo sanitizado (sem PII, < 500 caracteres); só as tools do
  especialista ativo são expostas; IDs/versões testados contra `PROMPTS.md`.
- **Tools** (`tools/banking.py`): `validate_client_cpf`, `authenticate_client`,
  `get_credit_limit`, `request_limit_increase`, `update_credit_score`,
  `get_exchange_rate`, `end_service`. Estado e serviços são injetados
  (`InjectedToolArg`) e ficam
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

- Perguntas sobre o atendimento são reconhecidas como dúvida, não como pedido
  de operação, e respondidas pelo catálogo de conhecimento; assunto fora do
  escopo é redirecionado com cordialidade.
- Autenticação com 3 tentativas e encerramento cordial. O pedido feito antes de
  autenticar fica guardado e é retomado assim que a autenticação conclui, sem
  pedir de novo o que o cliente acabou de dizer.
- Consulta de limite e solicitação de aumento com decisão por score; aprovação atualiza `clientes.csv`.
- Entrevista de crédito direta (pedido de score) ou após rejeição: o pedido de
  score abre com uma mensagem curta que explica os cinco itens, o uso das
  respostas e a ausência de garantia, e já faz a primeira pergunta; desistir no
  meio descarta tudo. Reanálise quando houver limite pendente.
- Estilo único em todos os agentes: no máximo três frases por resposta e sem
  travessão. A regra vale por prompt, por limite de schema na redação e por
  normalização determinística da saída do modelo.
- Cotação de moedas por nome (`dólar`, `euro`, `iene` etc.) ou par (`EUR-USD`),
  com tratamento humanizado de indisponibilidade e sem inventar valores.
- Dúvidas sobre como aumentar limite ou score orientam a entrevista de crédito;
  pedidos diretos de novo limite continuam sendo avaliados pelo score atual.
- A triagem separa consultar limite, aumentar limite e atualizar score casando a
  ação do cliente com o substantivo que ela atinge, e não por palavra solta:
  `saber meu limite antes de pedir aumento` é consulta, `aumentar meu limite
  porque o score melhorou` é aumento.
- Histórico completo permanece visível enquanto a aba estiver aberta; somente
  as seis mensagens mais recentes são enviadas ao grafo e ao Groq.
- Encerramento (`encerrar`, `sair`, `finalizar`…) prioritário em qualquer nó.
- Pergunta sobre o atendimento (`o que você pode fazer?`) respondida em
  qualquer nó, exceto dentro da entrevista.
- UI Streamlit em duas telas: a inicial centraliza o campo de mensagem e oferece
  quatro atalhos (visualizar limite, solicitar aumento de crédito, entrevista
  para atualizar crédito e cotação de moedas) que viram mensagem do cliente; uma
  transição animada abre o chat. No chat, cada solicitação pergunta se deseja
  continuar ou encerrar; após o encerramento, informar o CPF inicia outro
  atendimento. Sessão persistente, mascaramento de dados, tipografia Inter,
  cabeçalho simples, botão de novo atendimento, avatares por emoji, balões
  responsivos e temas claro/escuro consistentes.
- Auditoria técnica consultável por sessão + métricas por chamada de LLM.
- Mais de 200 testes (unitários, integração e E2E) + `docs/TEST_PLAN.md` de
  homologação.

## Desafios enfrentados e soluções

- **Pydantic validava dependências injetadas** e rejeitava fakes estruturais:
  `SkipValidation` nos serviços injetados, mantendo tipos concretos por escopo.
- **Invariante `ended`/`end_reason`**: atribuições separadas quebravam a
  validação — transição atômica em `ConversationState.end()`.
- **Texto livre com PII chegando ao LLM**: máscara de CPF, data, números e
  caracteres de estrutura + no máximo 5 mensagens anteriores mascaradas. A
  lista positiva de termos foi abandonada porque descartava a negação, e sem
  "não" o modelo nunca via uma recusa.
- **Cliente que desiste no meio de um fluxo**: cada especialista repetia a
  própria pergunta. Agora o passo é descartado e o pedido novo, quando existe,
  é entregue ao especialista certo no mesmo turno por uma aresta condicional.
- **LLM que extrai sem inventar**: "quero uns 8 mil" ou "ganho 5000 e sou
  registrado" só valem se cada número aparece no texto (dígito, "mil", "k" ou
  palavra de zero a dez) e cada valor repassa pelo parser do serviço; moeda e
  tópico só dentro dos conjuntos conhecidos; pergunta de esclarecimento só sem
  dígito e terminando em `?`. O que não aterra cai na pergunta canônica.
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

**Uso do LLM**: o Groq gera as boas-vindas por um prompt isolado, sem estado,
histórico ou tools. Na triagem, autenticação, encerramento e rotas claras usam
zero chamada; texto pós-autenticação ambíguo usa uma chamada de classificação,
com fallback determinístico. Dentro de um fluxo, texto que não é valor, moeda,
item da entrevista ou confirmação passa pelo parser de recusa e, se ele não
resolver, por uma leitura do turno (`understanding`) que traz recusa, pedido
novo, valor ("uns 8 mil"), par de moedas, respostas da entrevista ditas de uma
vez, tópico de dúvida e uma pergunta de esclarecimento. Cada campo só vale
depois de aterrado no texto do cliente: o Groq entende, o Python decide. Essa
leitura é feita uma vez por turno e compartilhada pelos nós. Crédito, Entrevista e Câmbio usam uma
chamada por turno para redigir o texto canônico com fatos mascarados (até duas
no turno com classificação), recebendo junto a pergunta do cliente com PII
mascarada para responderem no tom de quem perguntou. CPF, data, números,
sim/não, cálculos, encerramento e autenticação continuam determinísticos. Temperatura `0.3`, saída
de 500 tokens e timeout de 30 s; saída inválida ou falha preserva integralmente
a resposta canônica.

**Conhecimento**: perguntas sobre o atendimento (por que um pedido foi recusado,
o que é score, se há cobrança, de onde vem a cotação) são respondidas a partir de
um catálogo curado em `src/banco_agil/knowledge/catalog.py`, recuperado por
sobreposição de termos com `BaseRetriever` do LangChain; quando os termos não
bastam, o Groq aponta o tópico e o código confere que ele existe. A resposta
ganha fatos determinísticos do cliente (score atual e teto da faixa, lidos do
repositório) entre a explicação e a pergunta final. O catálogo explica
política e nunca calcula: limite, score e cotação continuam vindo das tools. Sem
correspondência, o atendimento admite que não sabe em vez de inventar.

**Limitações**: sem banco vetorial nem embeddings (a recuperação é por termo sobre
um catálogo pequeno e curado, o que dispensa índice vetorial e mantém o resultado
auditável); câmbio exige rede; LLM nunca decide aprovação, limite, score ou
cotação, apenas sugere a rota ambígua e redige o canônico.

## Tutorial de execução e testes

Pré-requisitos: Python 3.12+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edite .env e preencha BANCO_AGIL_GROQ_API_KEY para ativar o Groq
```

Interface (dados fictícios: CPF `11144477735`, nascimento `20/05/1990`):

```bash
streamlit run app.py
```

Sem `BANCO_AGIL_GROQ_API_KEY`, a aplicação roda em **modo determinístico**: nesse
modo, nenhuma chamada ao provedor é realizada. Com Groq ativo, cada especialista cria
primeiro uma resposta canônica a partir das regras e tools em Python. O modelo
gera a apresentação inicial e redige as respostas de Crédito, Entrevista e
Câmbio a partir do canônico com fatos mascarados, sem alterar fatos, valores ou
decisões. Junto do canônico ele recebe a pergunta do cliente — sem CPF,
nascimento, números ou caracteres de estrutura — para reconhecer o pedido e
responder com as palavras de quem perguntou; a pergunta orienta o tom, nunca o
conteúdo, e a saída só é aceita se preservar marcadores, números, decisão e
pergunta do canônico. Todo nó usa parser determinístico primeiro e só classifica
via Groq quando o texto continua ambíguo, seja a intenção na triagem, seja uma
recusa ou troca de assunto no meio de um fluxo; cada turno faz no máximo uma
chamada de classificação e uma de redação.

Roteiro na UI: na tela inicial, clique em **Visualizar limite** (ou digite o
pedido no campo central) → informe o CPF → informe o nascimento →
`qual é meu limite?`
(`R$ 2.500,00`) → `quero aumentar meu limite` → `4000` → responda `encerrar`
para finalizar; informar o CPF inicia outro atendimento. Demonstração completa
em `docs/DEMO.md`; homologação em `docs/TEST_PLAN.md`.

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
