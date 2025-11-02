# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app itself
COPY . .

# Streamlit runs on the port Cloud Run gives via $PORT
ENV PORT=8080
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080
CMD ["streamlit", "run", "streamlit_app.py"]
