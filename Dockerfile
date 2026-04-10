FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Health check for HF Spaces
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Start the FastAPI server via openenv's create_app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
