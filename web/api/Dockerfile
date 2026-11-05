# API веб-карты: только ядро пакета и extra [web], без геостека обучения.
# Сборка из корня репозитория: docker compose --profile web build api
FROM python:3.10.16-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY requirements/py310-geo.txt requirements/py310-web.txt requirements/
COPY src ./src
COPY web/api ./web/api
RUN pip install -c requirements/py310-web.txt ".[web]" \
    && useradd --system --uid 10001 jerqazyna
ENV PYTHONPATH=/app/web/api
USER jerqazyna
EXPOSE 8000
CMD ["uvicorn", "jerqazyna_web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
