FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (avoids pip warnings)
RUN useradd -m appuser
WORKDIR /app
COPY --chown=appuser:appuser . .
USER appuser

# Install Python dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Use Render's dynamic PORT
ENV PORT=10000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}"]
