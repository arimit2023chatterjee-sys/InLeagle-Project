FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/app/hf_cache
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('bhavyagiri/InLegal-Sbert', cache_folder='/app/hf_cache')"

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]