# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY streamlit_app__2_.py .
COPY position_calculator.py .
COPY interstitial_engine.py .

# Expose Streamlit port
EXPOSE 8080

# Configure Streamlit
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app__2_.py"]
