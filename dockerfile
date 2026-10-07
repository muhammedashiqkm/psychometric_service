FROM python:3.11-slim-bullseye

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN useradd -m appuser

RUN mkdir -p /code/logs && chown -R appuser:appuser /code/logs

COPY --chown=appuser:appuser app/ ./app/

USER appuser

# The port comes from .env. Shell form on purpose: the exec form would hand
# uvicorn the literal text ${PORT} instead of the number.
ENV PORT=5010
EXPOSE ${PORT}

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-5010} --workers 2