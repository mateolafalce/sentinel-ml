FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (no project itself yet)
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src/ ./src/
COPY static/ ./static/

# Expose Flask port
EXPOSE 5000

# Run the server
CMD ["uv", "run", "python", "-m", "src.server"]
