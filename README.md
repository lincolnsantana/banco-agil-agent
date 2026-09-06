# Banco Ágil — Chat bancário com especialistas internos

<p align="center">
  <img alt="Python 3.12 ou superior" src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit na interface" src="https://img.shields.io/badge/Streamlit-interface-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="LangGraph na orquestração" src="https://img.shields.io/badge/LangGraph-orquestração-1C3C3C?logo=langgraph&logoColor=white">
  <img alt="LangChain Core" src="https://img.shields.io/badge/LangChain-core-1C3C3C?logo=langchain&logoColor=white">
  <img alt="Groq como provedor de LLM" src="https://img.shields.io/badge/Groq-LLM-F55036">
  <img alt="Pydantic v2 na validação" src="https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white">
  <img alt="HTTPX como cliente REST" src="https://img.shields.io/badge/HTTPX-cliente_REST-2C5BB4">
  <img alt="AwesomeAPI como fonte das cotações" src="https://img.shields.io/badge/AwesomeAPI-cotações-0A7EA4">
  <img alt="SQLite na auditoria" src="https://img.shields.io/badge/SQLite-auditoria-003B57?logo=sqlite&logoColor=white">
  <img alt="Pytest com 584 testes" src="https://img.shields.io/badge/Pytest-584_testes-0A9EDC?logo=pytest&logoColor=white">
  <img alt="Ruff no lint e no format" src="https://img.shields.io/badge/Ruff-lint_e_format-D7FF64?logo=ruff&logoColor=black">
  <img alt="Mypy em modo strict" src="https://img.shields.io/badge/Mypy-strict-2A6DB2">
  <img alt="Docker e Docker Compose" src="https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white">
</p>

## Visão Geral

O Banco Ágil é um banco digital fictício cujo atendimento ao cliente é feito por
quatro especialistas internos de IA, apresentados ao cliente como uma única
conversa contínua em Streamlit:

- **Triagem**: explica a validação, verifica imediatamente o CPF no cadastro e
  pede novamente quando ele é inválido. Para CPF localizado, solicita nascimento
  em `DD/MM/AAAA` e só então autentica; encerra após a terceira falha.
- **Crédito**: consulta o limite atual e processa pedidos de aumento. Situa o
  pedido no limite que o cliente tem hoje, decide pela faixa de score, registra
  todo pedido em `solicitacoes_aumento_limite.csv` e, quando aprovado, atualiza
  o limite em `clientes.csv`.
- **Entrevista de Crédito**: com consentimento, coleta 5 dados financeiros, um
  por vez, recalcula o score por fórmula determinística, atualiza
  `clientes.csv` e devolve ao Crédito para reanálise.
- **Câmbio**: consulta a cotação atual via API REST/JSON (AwesomeAPI), informa
  valor e horário da última atualização e fecha convidando a outro serviço;
  nunca inventa valor nem recomenda investimento.

Além dos quatro exigidos pelo desafio, um quinto especialista interno responde
**dúvidas sobre o próprio atendimento** (por que um pedido foi recusado, o que é
score, se há cobrança) a partir de um catálogo curado, sem calcular nem decidir
nada.

![Demonstração do atendimento: da tela inicial ao chat, autenticação por CPF e
nascimento, pedido de aumento situado no limite atual, aprovação de R$ 2.500,00
para R$ 4.500,00 e o painel de perguntas rápidas acima do campo de
texto](docs/demo.gif)

**Demonstração ao vivo:** <https://COLE-AQUI-A-URL-DO-APP.streamlit.app>
<!-- Troque a URL acima pela do app publicado no Streamlit Community Cloud. -->

Tudo roda offline nos testes (mocks + fixtures temporárias). Nenhuma credencial
real é necessária; sem chave do provedor, os fluxos determinísticos funcionam e
intenções ambíguas recebem pedido de esclarecimento.

## Estrutura do projeto

```text
.
├── app.py                          interface Streamlit: telas, sessão e render
├── pyproject.toml                  dependências, Ruff, Mypy, Pytest e cobertura
├── Dockerfile, compose.yaml        execução em contêiner (opcional)
├── .env.example                    variáveis do provedor, dados e auditoria
├── .streamlit/config.toml          tema da interface
├── PROMPTS.md                      system prompts versionados, com IDs e limites
├── data/
│   ├── clientes.csv                10 clientes fictícios (CPF, nascimento, limite, score)
│   ├── score_limite.csv            faixas de score e teto de limite de cada uma
│   └── solicitacoes_aumento_limite.csv   uma linha por pedido de aumento
├── docs/
│   ├── DEMO.md                     roteiro de apresentação
│   └── TEST_PLAN.md                homologação manual passo a passo
├── src/banco_agil/
│   ├── config.py                   Settings tipado, lido do .env
│   ├── agents/                     orquestração da conversa
│   │   ├── graph.py                montagem do LangGraph e checkpoint de sessão
│   │   ├── router.py               rotas condicionais e limite de passos
│   │   ├── state.py                ConversationState validado do atendimento
│   │   ├── triage.py               autentica e identifica a necessidade
│   │   ├── credit.py               consulta de limite e pedido de aumento
│   │   ├── credit_interview.py     coleta consentida dos cinco dados
│   │   ├── exchange.py             cotação por nome da moeda ou par
│   │   ├── knowledge.py            dúvidas sobre o próprio atendimento
│   │   ├── understanding.py        leitura do turno pelo LLM, aterrada no texto
│   │   └── _shared.py              parsers, encerramento e redação final
│   ├── domain/                     modelos, enums e erros, sem framework
│   ├── services/                   regra de negócio por caso de uso
│   │   ├── authentication.py       tentativas, CPF e nascimento
│   │   ├── credit.py               avaliação do aumento pela faixa de score
│   │   ├── credit_interview.py     progresso da entrevista e novo score
│   │   ├── score.py                cálculo puro do score
│   │   ├── exchange.py             caso de uso da cotação
│   │   ├── knowledge.py            recuperação no catálogo curado
│   │   ├── welcome.py              saudação gerada pelo modelo
│   │   └── conversation.py         executa um turno do grafo
│   ├── repositories/               persistência: CSV do negócio e SQLite da auditoria
│   ├── integrations/               adaptadores externos: Groq e AwesomeAPI
│   ├── tools/banking.py            as sete tools expostas aos especialistas
│   ├── prompts/                    templates, registro de versões e renderização
│   ├── knowledge/catalog.py        catálogo curado de perguntas sobre o atendimento
│   └── observability/              logs sem PII e métricas de LLM
└── tests/
    ├── unit/                       regras puras, parsers e estado da UI
    ├── integration/                grafo, prompts, CSV, SQLite e HTTP falso
    ├── e2e/                        jornadas completas offline
    └── evals/                      avaliação de intenção e latência dos prompts
```

## Arquitetura, agentes, fluxos e manipulação de dados

```text
UI (app.py) -> ConversationService -> Graph (LangGraph) -> Tools -> Services
Services -> protocolos de repository/integration -> CSV / SQLite / HTTP
```

- **UI** (`app.py`): tela inicial com campo central e quatro atalhos, transição
  animada para o chat, conversa que abre pela mensagem do cliente e responde com
  as boas-vindas e o pedido de CPF, painel com três perguntas rápidas acima do
  campo de texto sempre que um especialista conclui o serviço, sessão e histórico
  entre reruns, máscara de CPF/nascimento na exibição, erros recuperáveis
  genéricos, atalho de volta à tela inicial e botão de novo atendimento quando
  a conversa encerra. Sem regra de negócio.
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

## Tutorial de execução e testes

### Pré-requisitos

| Requisito | Versão | Observação |
| --- | --- | --- |
| Python | 3.12 ou superior | única exigência obrigatória |
| pip | acompanha o Python | instala o projeto e as dependências |
| Git | qualquer | para clonar o repositório |
| Chave do Groq | opcional | sem ela a aplicação roda em modo determinístico |
| Docker | opcional | alternativa ao ambiente local |

### Instalação

**Linux e macOS**

```bash
git clone https://github.com/lincolnsantana/desafio-tecnico-ia.git
cd desafio-tecnico-ia

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/lincolnsantana/desafio-tecnico-ia.git
cd desafio-tecnico-ia

py -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

**Windows (Prompt de Comando)**

```bat
git clone https://github.com/lincolnsantana/desafio-tecnico-ia.git
cd desafio-tecnico-ia

py -m venv .venv
.venv\Scripts\activate.bat

pip install -e ".[dev]"
```

Se o PowerShell recusar o `Activate.ps1` por política de execução, rode uma vez
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` na mesma janela, ou
use o Prompt de Comando.

O `pip install -e ".[dev]"` é o único comando de instalação: ele lê o
`pyproject.toml` e traz **todas** as dependências, inclusive o Streamlit. Não é
preciso instalar nenhuma delas separadamente.

| Pacote | Para que serve |
| --- | --- |
| `streamlit` | interface do chat |
| `langgraph` | grafo que orquestra os especialistas |
| `langchain-core` | mensagens, tools e contratos do LLM |
| `langchain-groq` | adaptador do provedor de LLM |
| `pydantic`, `pydantic-settings` | modelos validados e leitura do `.env` |
| `httpx` | cliente REST que consulta a cotação |
| `filelock` | escrita concorrente segura nos CSV |
| `pytest`, `pytest-cov`, `respx` | testes e HTTP falso (extra `[dev]`) |
| `ruff`, `mypy` | formatação, lint e tipos (extra `[dev]`) |

Com o ambiente virtual ativo, os comandos a seguir são iguais nos três sistemas.
Confirme que deu certo:

```bash
streamlit --version                       # Streamlit, version 1.63.0
python -c "import banco_agil; print('ok')"
```

### Configuração

Copie o arquivo de exemplo e, se quiser a redação pelo modelo, preencha
`BANCO_AGIL_GROQ_API_KEY`.

**Linux e macOS**

```bash
cp .env.example .env
```

**Windows (PowerShell)**

```powershell
Copy-Item .env.example .env
```

**Windows (Prompt de Comando)**

```bat
copy .env.example .env
```

### Rodar localmente

```bash
streamlit run app.py
```

O Streamlit sobe em <http://localhost:8501> e abre o navegador sozinho. Para
encerrar, `Ctrl+C` no terminal.

### Clientes para teste

A base tem dez clientes fictícios, cobrindo as cinco faixas de score.
O limite de cada um respeita o teto da sua faixa, e a coluna final indica o que
cada perfil exercita:

| CPF | Nascimento | Limite | Score | Teto da faixa | Serve para testar |
| --- | --- | --- | --- | --- | --- |
| `11144477735` | `20/05/1990` | R$ 2.500,00 | 700 | R$ 10.000,00 | roteiro padrão: aumento até o teto é aprovado |
| `12345678909` | `30/11/1985` | R$ 5.000,00 | 800 | R$ 10.000,00 | folga confortável dentro da faixa |
| `52998224725` | `14/03/1978` | R$ 500,00 | 180 | R$ 1.000,00 | score baixo: quase todo pedido é recusado |
| `39053344705` | `02/09/1996` | R$ 1.200,00 | 350 | R$ 2.500,00 | faixa intermediária baixa |
| `87531694255` | `07/12/1969` | R$ 2.500,00 | 480 | R$ 2.500,00 | já no teto: recusa e oferta de entrevista |
| `46821975337` | `25/06/2003` | R$ 3.000,00 | 520 | R$ 5.000,00 | cliente jovem no meio da faixa |
| `73186420571` | `18/01/1954` | R$ 4.800,00 | 690 | R$ 5.000,00 | quase no teto, na borda da faixa |
| `29461735855` | `30/07/1988` | R$ 9.800,00 | 845 | R$ 10.000,00 | borda superior da faixa |
| `61748392573` | `11/11/1992` | R$ 12.000,00 | 860 | R$ 20.000,00 | faixa mais alta |
| `15837264973` | `09/04/1975` | R$ 19.500,00 | 990 | R$ 20.000,00 | teto máximo do cadastro |

Os CPFs são fictícios, mas têm dígitos verificadores válidos. A entrevista de
crédito altera o score e o limite do cliente usado, então o `git checkout
data/clientes.csv` devolve a base ao estado inicial.

### Com e sem chave do provedor

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

### Roteiro na interface

Na tela inicial, clique em **Visualizar limite** (ou digite o
pedido no campo central) → informe o CPF → informe o nascimento →
`qual é meu limite?`
(`R$ 2.500,00`) → `quero aumentar meu limite` → `4000` → responda `encerrar`
para finalizar; o botão **Iniciar novo atendimento** começa outro do zero.
Demonstração completa em `docs/DEMO.md`; homologação em `docs/TEST_PLAN.md`.

### Testes e validação

```bash
ruff format --check .
ruff check .
mypy src
pytest --cov=src/banco_agil --cov-report=term-missing
pytest -q tests/integration tests/e2e
```

Os testes rodam offline: sem rede, sem credencial e sobre fixtures temporárias,
então nenhum deles toca os CSV de `data/`.

### Docker e CI (opcional)

```bash
docker compose config      # valida o arquivo
docker compose build
docker compose up --build  # aplicação em contêiner
```

O workflow `.github/workflows/ci.yml` repete a mesma validação a cada push, sem
chaves e sem serviços externos.

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

A redação pelo modelo depende do orçamento de tokens. O `.env.example` traz 500
tokens, calibrados para o `llama-3.3-70b-versatile`. Modelos de raciocínio, como
os `gpt-oss`, gastam tokens pensando antes de emitir a saída estruturada: com
orçamento curto, ou com a conta perto do limite por minuto, o provedor devolve
geração vazia e a chamada falha. O atendimento continua correto, porque a
resposta canônica prevalece, mas sem a variação da redação. Se as respostas
parecerem sempre iguais, esse é o primeiro lugar a olhar: aumente
`BANCO_AGIL_LLM_MAX_TOKENS` ou volte ao modelo documentado.

## Funcionalidades implementadas

- Perguntas sobre o atendimento são reconhecidas como dúvida, não como pedido
  de operação, e respondidas pelo catálogo de conhecimento; assunto fora do
  escopo é redirecionado com cordialidade.
- Autenticação com 3 tentativas e encerramento cordial. O pedido feito antes de
  autenticar fica guardado e é retomado assim que a autenticação conclui, sem
  pedir de novo o que o cliente acabou de dizer.
- Consulta de limite e solicitação de aumento com decisão por score; a
  aprovação atualiza `clientes.csv`. O pedido é sempre situado em números: o
  limite de hoje ao perguntar quanto o cliente quer, o passo de um valor ao
  outro na aprovação e a comparação com o vigente na recusa.
- Entrevista de crédito direta (pedido de score) ou após rejeição: o pedido de
  score abre com uma mensagem curta que explica os cinco itens, o uso das
  respostas e a ausência de garantia, e já faz a primeira pergunta; desistir no
  meio descarta tudo. Reanálise quando houver limite pendente.
- Estilo único em todos os agentes: no máximo três frases por resposta e sem
  travessão. A regra vale por prompt, por limite de schema na redação e por
  normalização determinística da saída do modelo.
- Cotação de moedas por nome (`dólar`, `euro`, `iene` etc.) ou par (`EUR-USD`),
  com tratamento humanizado de indisponibilidade e sem inventar valores. A
  resposta traz valor e horário da atualização e convida a outro serviço.
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
  transição animada abre o chat. Concluído um serviço, o assistente convida a
  continuar ou a encerrar; encerrado o atendimento, um botão inicia outro do
  zero. Ao fim de cada serviço, um painel acima do campo oferece três
  perguntas rápidas do especialista que respondeu, sem bloquear a digitação.
  Sessão persistente, mascaramento de dados, tipografia Inter, atalho de volta
  à tela inicial, fala do especialista direto na página e balão só para o
  cliente, layout responsivo e temas claro/escuro consistentes.
- Auditoria técnica consultável por sessão + métricas por chamada de LLM.
- 584 testes (unitários, integração, E2E e evals) + `docs/TEST_PLAN.md` de
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
- **Base de testes que exercitasse as regras**: `data/clientes.csv` nasceu com
  um cliente só, o que não cobria recusa por faixa nem cliente já no teto. Hoje
  são dez perfis fictícios, um por situação relevante, com o limite de cada um
  dentro do teto da sua faixa de score.
