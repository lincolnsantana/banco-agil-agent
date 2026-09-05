"""Recuperacao do conteudo explicativo, por termo e sem banco vetorial.

Usa `Document` e `BaseRetriever` do `langchain-core`, que ja e dependencia do
projeto. Nao ha embedding nem indice vetorial: o catalogo tem poucas dezenas de
entradas curadas, em que sobreposicao de termos normalizados resolve melhor,
sem dependencia nova e com resultado auditavel.
"""

import re
import unicodedata
from collections.abc import Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from banco_agil.knowledge.catalog import KNOWLEDGE_CATALOG, KnowledgeEntry

MIN_TERM_MATCHES = 1


def _tokenize(value: str) -> frozenset[str]:
    """Reduz a pergunta a termos comparaveis, sem acento nem pontuacao.

    O servico normaliza por conta propria, como `welcome.py` ja faz: importar
    de `agents` inverteria a direcao das camadas.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return frozenset(re.sub(r"[^a-z0-9]+", " ", without_accents.casefold()).split())


def _matched_terms(entry: KnowledgeEntry, words: frozenset[str]) -> int:
    return len(entry.terms & words)


def rank_entries(
    user_text: str,
    catalog: Sequence[KnowledgeEntry] = KNOWLEDGE_CATALOG,
) -> list[KnowledgeEntry]:
    """Ordena as entradas pela sobreposicao de termos com a pergunta.

    Empate e desfeito pela ordem do catalogo, nao pela chave: a ordem e
    autoral e coloca a explicacao mais geral antes da mais especifica, de modo
    que "o que e score" ganhe de "como calculam o score". Ordenacao estavel,
    para o teste poder afirmar um resultado exato.
    """
    words = _tokenize(user_text)
    scored = [
        (_matched_terms(entry, words), position, entry)
        for position, entry in enumerate(catalog)
        if _matched_terms(entry, words) >= MIN_TERM_MATCHES
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [entry for _, _, entry in scored]


class KnowledgeService:
    """Encontra a explicacao mais proxima da pergunta do cliente."""

    def __init__(
        self,
        catalog: Sequence[KnowledgeEntry] = KNOWLEDGE_CATALOG,
    ) -> None:
        """Guarda o catalogo consultado, permitindo substituicao em teste."""
        self._catalog = tuple(catalog)

    def find(self, user_text: str) -> KnowledgeEntry | None:
        """Devolve a melhor explicacao ou None quando nada alcanca o limiar.

        Devolver None e deliberado: sem correspondencia, o especialista assume
        que nao sabe, em vez de responder com a entrada menos ruim.
        """
        ranked = rank_entries(user_text, self._catalog)
        return ranked[0] if ranked else None


class KnowledgeRetriever(BaseRetriever):
    """Expoe o catalogo pelo contrato de retriever do LangChain."""

    catalog: tuple[KnowledgeEntry, ...] = KNOWLEDGE_CATALOG
    top_k: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Converte as entradas mais proximas em documentos recuperados."""
        del run_manager
        return [
            Document(page_content=entry.answer, metadata={"key": entry.key})
            for entry in rank_entries(query, self.catalog)[: self.top_k]
        ]
