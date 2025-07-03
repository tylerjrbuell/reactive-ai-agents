# Reflex - Reactive Agents Framework
# Official Docker Image
FROM python:3.11-slim

# Set metadata
LABEL maintainer="Tyler Buell"
LABEL description="Reflex - A modern, reactive AI agent framework for intelligent task execution"
LABEL version="0.1.0a6"
LABEL org.opencontainers.image.source="https://github.com/tylerbuell/reactive-ai-agent"
LABEL org.opencontainers.image.description="Reflex - Reactive Agents Taking Action"
LABEL org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    REFLEX_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Create app directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-dev \
    && rm -rf $POETRY_CACHE_DIR

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create non-root user for security
RUN groupadd -r reflex && useradd -r -g reflex reflex
RUN chown -R reflex:reflex /app
USER reflex

# Create directories for data persistence
RUN mkdir -p /app/data /app/logs /app/config

# Expose port for potential web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD reflex version || exit 1

# Default command
CMD ["reflex", "--help"] 