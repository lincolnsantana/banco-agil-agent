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
  <img alt="Pytest com 624 testes" src="https://img.shields.io/badge/Pytest-624_testes-0A9EDC?logo=pytest&logoColor=white">
  <img alt="Ruff no lint e no format" src="https://img.shields.io/badge/Ruff-lint_e_format-D7FF64?logo=ruff&logoColor=black">
  <img alt="Mypy em modo strict" src="https://img.shields.io/badge/Mypy-strict-2A6DB2">
  <img alt="Docker e Docker Compose" src="https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white">
</p>

## Visão Geral

**Teste aqui:** <https://banco-agil-agent.streamlit.app/>

O Banco Ágil é um banco digital fictício. O atendimento é feito por
especialistas de IA com escopos separados, que o cliente enxerga como uma única
conversa em Streamlit:

- **Triagem** — autentica por CPF e nascimento e encaminha ao especialista certo.
- **Crédito** — mostra o limite atual e decide pedidos de aumento pela faixa de
  score.
- **Entrevista de Crédito** — com consentimento, coleta cinco dados financeiros,
  recalcula o score e devolve o pedido para nova análise.
- **Câmbio** — consulta a cotação do momento numa API pública, sem inventar
  valor nem recomendar investimento.
- **Conhecimento** — acréscimo aos quatro exigidos pelo desafio: explica o
  próprio atendimento a partir de um catálogo curado, sem calcular nem decidir.

![Demonstração do atendimento: da tela inicial ao chat, autenticação por CPF e
nascimento, pedido de aumento situado no limite atual, aprovação de R$ 2.500,00
para R$ 4.500,00 e o painel de perguntas rápidas acima do campo de
texto](docs/demo.gif)

A aplicação roda sem credencial nenhuma: sem chave do provedor, as decisões e os
fluxos são os mesmos, e só o texto das respostas deixa de variar. Os testes rodam
offline, sobre fixtures temporárias.

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

### O caminho de uma mensagem

```text
Cliente
   │
   ▼
UI ........................ app.py            telas, sessão e máscara de CPF
   │
   ▼
ConversationService ....... services/         um turno: estado + fala → resposta
   │
   ▼
Grafo ..................... agents/graph.py   escolhe o especialista do turno
   │
   ▼
Nós ....................... agents/*.py       Triagem, Crédito, Entrevista,
   │                                          Câmbio e Conhecimento
   ▼
Tools ..................... tools/banking.py  as sete operações permitidas
   │
   ▼
Serviços .................. services/*.py     a regra de negócio, sem framework
   │
   ▼
Repositórios e integrações                    CSV, SQLite e HTTP
```

Cada camada só conhece a de baixo, e o domínio não importa framework nenhum.

### Quem faz o quê

| Camada | Onde | Responsabilidade |
| --- | --- | --- |
| UI | `app.py` | Desenha as telas, guarda a sessão entre reruns e mascara CPF e nascimento na exibição. Nenhuma regra de negócio. |
| Grafo | `agents/graph.py`, `agents/router.py` | Liga os nós e impõe os limites do turno. |
| Nós | `agents/triage.py`, `credit.py`, `credit_interview.py`, `exchange.py`, `knowledge.py` | Um especialista por assunto. Coordenam estado e tools; não tocam em arquivo. |
| Prompts | `prompts/` | Um system message por turno: global + especialista ativo + estado mínimo sem PII. Versionados e testados contra `PROMPTS.md`. |
| Tools | `tools/banking.py` | As sete operações do atendimento. Estado e serviços são injetados e ficam fora do schema visível ao modelo; o CPF vem sempre do estado confiável. |
| Serviços | `services/` | Onde a decisão acontece: autenticação, faixa de score, cálculo do score, cotação. |
| Repositórios | `repositories/` | CSV do negócio e SQLite da auditoria, atrás de protocolos. |

### Como um turno é decidido

Toda mensagem passa pelas mesmas verificações, nesta ordem:

1. Pedido de encerramento, que vale em qualquer nó e a qualquer momento e por
   isso é a primeira coisa que cada especialista checa.
2. Autenticação. Enquanto o cliente não se identifica, só a triagem responde. O
   que ele pediu antes disso fica guardado e é respondido quando a autenticação
   termina.
3. Resposta ao passo em andamento, quando existe um: o valor do aumento, a
   moeda, o item da entrevista.
4. Rota, quando não há passo pendente. O parser tenta primeiro; se o texto
   continuar ambíguo, o Groq classifica, e o que ele entende precisa estar
   escrito na fala do cliente.

Dois limites protegem o turno: `MAX_HANDLER_STEPS=2` e `recursion_limit=8`
evitam que os nós fiquem se chamando, e o modelo nunca recebe mais do que as
seis mensagens mais recentes.

Uma pergunta no meio de um passo não conta como resposta nem como desistência. O
catálogo explica, o especialista repete a pergunta pendente e o que já foi
coletado continua lá.

### Onde os dados ficam

| Arquivo | Conteúdo | Quem escreve |
| --- | --- | --- |
| `data/clientes.csv` | CPF, nascimento, limite e score | aprovação de aumento e entrevista |
| `data/score_limite.csv` | faixas de 0 a 1000, sem lacunas | ninguém: só leitura |
| `data/solicitacoes_aumento_limite.csv` | uma linha por pedido, `pendente` → `aprovado`/`rejeitado` | crédito |
| `var/*.db` (opcional) | eventos do turno e métricas de LLM, sem PII | serviço de conversa |

Toda escrita passa por `filelock`, arquivo temporário e `os.replace`, então uma
falha no meio não deixa CSV pela metade. Valores são `Decimal` e horários são
UTC em ISO 8601. Falha de auditoria nunca interrompe o atendimento.

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
git clone https://github.com/lincolnsantana/banco-agil-agent.git
cd banco-agil-agent

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/lincolnsantana/banco-agil-agent.git
cd banco-agil-agent

py -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

**Windows (Prompt de Comando)**

```bat
git clone https://github.com/lincolnsantana/banco-agil-agent.git
cd banco-agil-agent

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

### Cota da API de câmbio

A AwesomeAPI é pública e funciona sem credencial, mas nesse modo a cota é contada
**por IP**. Em hospedagem de IP compartilhado, como o Streamlit Community Cloud,
o limite estoura por uso alheio e a cotação passa a responder
`429 QuotaExceeded` — o cliente vê "cotação indisponível" e o log registra o
status. Gere um token gratuito em <https://awesomeapi.com.br> e configure
`BANCO_AGIL_AWESOMEAPI_TOKEN` para a cota passar a ser da sua conta.

### Com e sem chave do provedor

Sem `BANCO_AGIL_GROQ_API_KEY` nenhuma chamada ao provedor acontece, e o
atendimento funciona igual. O que muda é só a escrita:

| | Sem chave | Com chave |
| --- | --- | --- |
| Decisões, valores e rotas claras | idênticas | idênticas |
| Texto das respostas | fixo | escrito pelo modelo, no tom do cliente |
| Frase ambígua | pede esclarecimento | é interpretada e validada em Python |

O mecanismo está em [Uso do LLM](#uso-do-llm).

### Roteiro na interface

Na tela inicial, clique em **Visualizar limite** ou digite o pedido no campo
central. Depois:

`CPF` → `nascimento` → `qual é meu limite?` (responde `R$ 2.500,00`) →
`quero aumentar meu limite` → `4000` (aprovado) → `encerrar`

O botão **Iniciar novo atendimento** começa outro do zero. Roteiro completo em
`docs/DEMO.md`; homologação em `docs/TEST_PLAN.md`.

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

### Uso do LLM

O modelo escolhe palavras e desempata leituras ambíguas. Ele nunca decide um
fato. São três tarefas, cada uma com sua guarda:

| Tarefa | O que o modelo faz | O que o impede de errar |
| --- | --- | --- |
| Boas-vindas | escreve a saudação de abertura | prompt isolado, sem estado, histórico ou tools |
| Entendimento | lê um turno ambíguo e devolve estrutura: intenção, valor, moeda, respostas da entrevista | cada campo precisa estar sustentado no texto do cliente, ou é descartado |
| Redação | escreve a fala final a partir do texto já validado | os fatos vão mascarados; marcadores, números e decisão são conferidos na volta |

Autenticação, encerramento, cálculo de score, faixa de limite e rotas claras não
passam pelo modelo. São no máximo duas chamadas por turno, temperatura `0.5`,
500 tokens e timeout de 30 s. Qualquer falha ou saída fora das guardas preserva
a resposta determinística.

### Conhecimento

Dúvidas sobre o atendimento vêm de um catálogo curado em
`src/banco_agil/knowledge/catalog.py`, recuperado por sobreposição de termos com
`BaseRetriever` do LangChain. Quando os termos não bastam, o Groq aponta o tópico
e o código confere que ele existe.

A explicação ganha os fatos do cliente lidos do repositório, como o score atual e
o teto da faixa. O catálogo explica política e nunca calcula: limite, score e
cotação continuam vindo das tools. Sem correspondência, o atendimento admite que
não sabe em vez de inventar.

### Limitações e diagnóstico

- Sem banco vetorial nem embeddings. A recuperação é por termo sobre um catálogo
  pequeno e curado, o que dispensa índice vetorial e mantém o resultado
  auditável.
- O câmbio exige rede, e a AwesomeAPI limita a cota por IP quando não há token.
- O modelo nunca decide aprovação, limite, score ou cotação.

**Se as respostas parecerem sempre iguais**, a redação está falhando em silêncio
e o texto determinístico está prevalecendo. Duas causas comuns, nesta ordem:

1. **O modelo configurado não existe mais.** Provedores aposentam modelos, e a
   chamada passa a falhar sem aviso. Confirme o que sua conta tem:

   ```bash
   curl -s https://api.groq.com/openai/v1/models \
     -H "Authorization: Bearer $BANCO_AGIL_GROQ_API_KEY" | grep '"id"'
   ```

2. **O orçamento de tokens é curto para o modelo escolhido.** Modelos de
   raciocínio, como os `gpt-oss`, gastam tokens pensando antes de produzir a
   saída estruturada; com pouco espaço, devolvem geração vazia. Os 500 tokens do
   `.env.example` são calibrados para o `qwen/qwen3.8-27b`.

## Funcionalidades implementadas

**Atendimento**

- Autenticação em duas etapas, com três tentativas e encerramento cordial.
- Consulta de limite e pedido de aumento decidido pela faixa de score. O pedido
  é sempre situado em números: o limite de hoje ao perguntar quanto o cliente
  quer, o passo de um valor ao outro na aprovação, a comparação com o vigente
  na recusa.
- Entrevista de crédito, pedida direto ou oferecida após uma recusa. Explica os
  cinco itens antes de começar, recalcula o score e devolve ao crédito para
  reanálise. Desistir no meio descarta tudo.
- Cotação por nome da moeda (`dólar`, `euro`, `iene`) ou por par (`EUR-USD`),
  com valor e horário da atualização, e indisponibilidade tratada sem inventar
  número.
- Dúvidas sobre o atendimento respondidas por um catálogo curado: por que um
  pedido foi recusado, o que é score, se há cobrança, de onde vem a cotação.

**Conversa**

- Pergunta é reconhecida como dúvida, e não como pedido de operação.
- O pedido feito antes de autenticar é retomado depois, sem repetir a pergunta.
- Dúvida no meio de um passo é respondida sem descartar o que já foi coletado.
- Encerrar funciona em qualquer momento, inclusive por despedida ("tchau",
  "era só isso").
- Troca de assunto no meio de um fluxo vai ao especialista certo no mesmo
  turno, em vez de repetir a pergunta anterior.
- Estilo único: no máximo três frases por resposta, sem travessão. A regra vale
  por prompt, por limite de schema e por normalização da saída do modelo.

**Interface**

- Duas telas: a inicial com campo central e quatro atalhos; o chat abre com uma
  transição animada.
- Ao fim de cada serviço, três perguntas rápidas do especialista que respondeu,
  acima do campo, sem bloquear a digitação.
- Encerrado o atendimento, um botão inicia outro do zero.
- CPF e nascimento mascarados na tela, temas claro e escuro, layout responsivo.

**Qualidade**

- Auditoria por sessão e métricas por chamada de LLM, sem PII.
- 624 testes (unitários, integração, E2E e evals), todos offline, mais o
  `docs/TEST_PLAN.md` de homologação manual.

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
