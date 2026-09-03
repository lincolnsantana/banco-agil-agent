# Imagem de execucao do atendimento Banco Agil (Streamlit).
# Sem segredos: a chave do provedor LLM, quando usada, chega via ambiente
# no runtime e nunca e copiada para a imagem.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/banco-agil

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml ./
COPY src/ ./src/
COPY app.py ./
COPY data/ ./data/

RUN pip install --no-cache-dir . \
    && mkdir -p data var \
    && chown -R appuser:appuser /srv/banco-agil

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status == 200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
