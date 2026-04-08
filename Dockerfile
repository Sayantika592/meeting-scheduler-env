FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (cached layer — rebuilds only if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# The server listens on port 8000
EXPOSE 8000

# Start the FastAPI server
# --host 0.0.0.0 makes it accessible from outside the container
# --port 8000 matches the EXPOSE above
CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
