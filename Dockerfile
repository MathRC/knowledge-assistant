# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from local app/ into container's /app
COPY app/ ./

# Copy configuration and data
COPY config.yaml .
COPY data/ ./data/

# Expose Streamlit port
EXPOSE 8501

# Run ingestion then launch Streamlit
CMD ["sh", "-c", "python ingest_data.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]