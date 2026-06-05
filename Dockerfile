# ============================================================
# DOCKERFILE — Store Sales Forecast API
# ============================================================

FROM python:3.10.13-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

FROM base AS runtime

COPY src/         ./src/
COPY serving/     ./serving/
COPY services/    ./services/
COPY forecast_models/ ./forecast_models/

RUN mkdir -p logs

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "serving.forecast_api:app", "--host", "0.0.0.0", "--port", "8000"]
