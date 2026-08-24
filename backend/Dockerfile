# --- Phase 4: Containerize the Application ---
# Builds a portable image so the app runs identically in any environment.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
