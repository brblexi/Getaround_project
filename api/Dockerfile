FROM python:3.11-slim

# HF Spaces route le trafic vers le port 7860 par défaut
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# la commande de lancement --> uvicorn sert l'app FastAPI ; --host 0.0.0.0 pour être joignable depuis l'extérieur
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
