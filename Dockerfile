FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends p7zip-full \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .

ENV KODA_SERVER_HOST=0.0.0.0
ENV KODA_SERVER_PORT=8765
ENV KODA_PROJECT_DIR=/app

EXPOSE 8765

CMD ["koda-pipeline"]
