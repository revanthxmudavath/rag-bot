# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    RUN_MODE=api \
    LOG_TO_STDOUT=true
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY app ./app
COPY bot ./bot
COPY data_science ./data_science
COPY prompts.txt ./prompts.txt
COPY run.py ./run.py
COPY requirements.txt ./requirements.txt

RUN useradd --create-home appuser \
    && mkdir -p logs .cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD sh -c "if [ \"$RUN_MODE\" = \"api\" ]; then curl -f http://localhost:8000/api/health >/dev/null 2>&1 || exit 1; else exit 0; fi"

CMD ["python", "run.py"]
