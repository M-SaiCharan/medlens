# Serving image (CPU only, no torch). Build after preparing data + model:
#   python scripts/download_data.py && python -m medlens.catalog
#   (train on GPU, copy models/retriever/ here) -- optional; falls back to fuzzy matching
#   docker build -t medlens . && docker run -p 7860:7860 medlens
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir $(grep -vE "^(#|$|kaggle|pytest|httpx|jupyter|matplotlib)" requirements.txt)

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --no-deps .
COPY --chown=user app ./app
COPY --chown=user data/processed/medicines.parquet ./data/processed/
COPY --chown=user data/raw/ddinter ./data/raw/ddinter
COPY --chown=user models ./models

USER user
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
