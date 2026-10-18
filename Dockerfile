FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système pour PyTorch / ChromaDB / compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source et des fichiers RAG
COPY . .

# Port standard imposé par Hugging Face Spaces
ENV PORT=7860
EXPOSE 7860

# Lancement avec Gunicorn (1 worker suffit pour éviter de doubler la RAM des modèles)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "app:app"]