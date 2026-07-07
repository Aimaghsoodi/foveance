# Reproducible Foveance environment (offline core + benchmark).
FROM python:3.12-slim

WORKDIR /app

# Install the package
RUN pip install --no-cache-dir foveance

# Proxy launch
ENTRYPOINT ["foveance", "proxy", "--host", "0.0.0.0", "--port", "8799"]
