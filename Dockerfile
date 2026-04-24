FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api /app/api
COPY notebooks /app/notebooks
RUN mkdir -p /app/checkpoints

EXPOSE 4000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "4000"]
