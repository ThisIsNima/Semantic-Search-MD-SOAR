# Pull base image
FROM python:3.12.0-slim

RUN apt-get update && \
    apt-get clean

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /opt/semantic-search

# Install dependencies
COPY vectors.csv .
COPY MCP/MCP-semanticSearch2.py .
COPY pyproject.toml .
RUN pip install -e .

# PORT
EXPOSE 8000

# Commands to run migration and start the server
CMD ["sh", "-c", "python MCP-semanticSearch2.py"]
