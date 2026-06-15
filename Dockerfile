# aicheckword FastAPI 集成服务（单 worker；初稿/审核 job 在进程内存）
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

WORKDIR /w

COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements-api.txt

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    CHROMA_PERSIST_DIR=/app/knowledge_store \
    UPLOADS_DIR=/app/uploads \
    TRAINING_DOCS_DIR=/app/training_docs \
    LIBREOFFICE_PATH=/usr/bin/soffice

WORKDIR /app

RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libreoffice-writer-nogui \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/bash app

COPY --from=builder /install /usr/local

COPY . .

RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /app/knowledge_store /app/uploads /app/training_docs \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
