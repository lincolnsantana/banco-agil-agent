"""Conteudo explicativo do atendimento, curado e versionado com o codigo.

O catalogo explica politica; nunca calcula nem decide. Limite, score e cotacao
continuam vindo das tools deterministicas.

As respostas obedecem tres invariantes verificados por teste, ditados pelas
guardas de aceite da humanizacao em `agents/_shared.py`:

- ate `MAX_ANSWER_LENGTH` caracteres, para sobrar folga dentro do teto da
  reescrita;
- sem digito, porque a guarda exige que os numeros da reescrita sejam
  subconjunto dos do canonico;
- terminando em pergunta, para a paridade de interrogacao ser estavel.
"""

from dataclasses import dataclass

from banco_agil.domain.enums import Intent

MAX_ANSWER_LENGTH = 180


@dataclass(frozen=True)
class KnowledgeEntry:
    """Explicacao curta associada aos termos que a tornam relevante."""

    key: str
    terms: frozenset[str]
    answer: str
    follow_up: Intent | None = None


KNOWLEDGE_CATALOG: tuple[KnowledgeEntry, ...] = (
    # --- Score e limite -------------------------------------------------
    KnowledgeEntry(
        key="what_is_score",
        terms=frozenset(
            {"score", "pontuacao", "significa", "nota", "bom", "boa", "ruim", "alto"}
        ),
        answer=(
            "Score é a nota que resume seu histórico de crédito e define a "
            "faixa de limite que posso liberar. Quer consultar seu limite "
            "atual?"
        ),
        follow_up=Intent.CREDIT_LIMIT,
    ),
    KnowledgeEntry(
        key="how_score_is_calculated",
        terms=frozenset({"calculam", "calcula", "calculado", "formula", "score"}),
        answer=(
            "Seu score vem de renda, tipo de emprego, despesas, dependentes e "
            "dívidas ativas. É a entrevista que atualiza esses dados. Quer "
            "fazer a entrevista?"
        ),
        follow_up=Intent.CREDIT_INTERVIEW,
    ),
    KnowledgeEntry(
        key="score_defines_limit",
        terms=frozenset({"faixa", "teto", "maximo", "define", "limite", "score"}),
        answer=(
            "Cada faixa de score tem um limite máximo. Seu pedido é aprovado "
            "quando cabe na faixa em que seu score está hoje. Quer ver seu "
            "limite atual?"
        ),
        follow_up=Intent.CREDIT_LIMIT,
    ),
    KnowledgeEntry(
        key="why_rejected",
        terms=frozenset({"rejeitado", "negado", "recusado", "reprovado"}),
        answer=(
            "Um pedido é recusado quando passa do teto da sua faixa de score "
            "atual. A entrevista pode atualizar o score e permitir nova "
            "análise. Quer tentar?"
        ),
        follow_up=Intent.CREDIT_INTERVIEW,
    ),
    KnowledgeEntry(
        key="can_ask_again",
        terms=frozenset({"denovo", "novamente", "outra", "depois", "repetir"}),
        answer=(
            "Pode pedir de novo quando quiser. Cada pedido é avaliado pelo "
            "score do momento, então atualizar o score costuma ajudar. Quer "
            "fazer a entrevista?"
        ),
        follow_up=Intent.CREDIT_INTERVIEW,
    ),
    KnowledgeEntry(
        key="analysis_time",
        terms=frozenset({"demora", "tempo", "prazo", "quando", "rapido"}),
        answer=(
            "A análise é na hora: assim que você informa o limite desejado, eu "
            "comparo com sua faixa e respondo aqui mesmo. Quer pedir um "
            "aumento agora?"
        ),
        follow_up=Intent.LIMIT_INCREASE,
    ),
    KnowledgeEntry(
        key="increase_fee",
        terms=frozenset({"taxa", "custa", "cobram", "cobra", "tarifa", "gratis"}),
        answer=(
            "Não há cobrança para pedir aumento nem para fazer a entrevista de "
            "crédito. Quer pedir um aumento?"
        ),
        follow_up=Intent.LIMIT_INCREASE,
    ),
    # --- Regras do atendimento ------------------------------------------
    KnowledgeEntry(
        key="why_authenticate",
        terms=frozenset({"autenticacao", "autenticar", "cpf", "seguranca", "validar"}),
        answer=(
            "Peço CPF e nascimento antes de tudo para confirmar que estou "
            "falando com o titular da conta. Podemos seguir?"
        ),
    ),
    KnowledgeEntry(
        key="interview_data_use",
        terms=frozenset({"dados", "guardam", "guarda", "privacidade", "armazenam"}),
        answer=(
            "As respostas da entrevista servem só para recalcular seu score. "
            "Se você desistir no meio, nada é guardado. Quer começar?"
        ),
        follow_up=Intent.CREDIT_INTERVIEW,
    ),
    KnowledgeEntry(
        key="interview_scope",
        terms=frozenset({"entrevista", "perguntas", "pergunta", "quantas"}),
        answer=(
            "A entrevista tem cinco perguntas sobre renda, emprego, despesas, "
            "dependentes e dívidas, e você pode parar quando quiser. Quer "
            "começar?"
        ),
        follow_up=Intent.CREDIT_INTERVIEW,
    ),
    # --- Cambio ----------------------------------------------------------
    KnowledgeEntry(
        key="exchange_source",
        terms=frozenset({"cotacao", "cambio", "fonte", "atualizada", "vem"}),
        answer=(
            "A cotação vem de uma fonte pública de mercado, consultada no "
            "momento da pergunta, e muda ao longo do dia. Quer consultar "
            "alguma moeda?"
        ),
        follow_up=Intent.EXCHANGE_RATE,
    ),
    KnowledgeEntry(
        key="currency_pair",
        terms=frozenset({"par", "moedas", "sigla", "usd", "eur", "brl"}),
        answer=(
            "O par diz qual moeda vale quanto na outra, como dólar em reais. "
            "Pode dizer só o nome da moeda. Quer ver alguma cotação?"
        ),
        follow_up=Intent.EXCHANGE_RATE,
    ),
    KnowledgeEntry(
        key="no_investment_advice",
        terms=frozenset({"investir", "comprar", "vender", "vale", "recomenda"}),
        answer=(
            "Não posso recomendar compra, venda ou investimento. Informo a "
            "cotação do momento e a decisão fica com você. Quer consultar uma "
            "moeda?"
        ),
        follow_up=Intent.EXCHANGE_RATE,
    ),
    # --- Fora de escopo, com saida cordial -------------------------------
    KnowledgeEntry(
        key="out_of_scope_billing",
        terms=frozenset({"fatura", "boleto", "pagamento", "vencimento", "atraso"}),
        answer=(
            "Fatura e pagamento não passam por mim. Aqui cuido de limite, "
            "aumento, entrevista de crédito e cotação de moedas. Posso ajudar "
            "com algum deles?"
        ),
    ),
    KnowledgeEntry(
        key="out_of_scope_fees",
        terms=frozenset({"anuidade", "seguro", "assinatura", "mensalidade"}),
        answer=(
            "Anuidade e seguro ficam com outra área. Comigo você resolve "
            "limite, aumento, entrevista de crédito e cotação de moedas. Qual "
            "deles quer ver?"
        ),
    ),
    KnowledgeEntry(
        key="out_of_scope_products",
        terms=frozenset({"emprestimo", "financiamento", "pix", "transferencia"}),
        answer=(
            "Esse serviço não faz parte deste atendimento. Posso cuidar de "
            "limite, aumento, entrevista de crédito e cotação de moedas. Quer "
            "algum desses?"
        ),
    ),
)
