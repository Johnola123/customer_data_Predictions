# Simple Dockerfile for the FastAPI churn service
FROM python:3.11-slim

WORKDIR /app

# Install system deps if needed
RUN pip install --no-cache-dir --upgrade pip

# Copy app and model
COPY app /app/app
COPY models /app/models
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
