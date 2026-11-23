FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 01_finetuned_rag_assistant/src/ ./01_finetuned_rag_assistant/src/
COPY 01_finetuned_rag_assistant/knowledge_base/ ./01_finetuned_rag_assistant/knowledge_base/

ENV PYTHONPATH=/app/01_finetuned_rag_assistant
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "-m", "src.app"]