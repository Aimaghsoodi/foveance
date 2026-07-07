# Reproducible Foveance environment (offline core + benchmark).
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Install the package with dev + bench extras; core itself is dependency-free.
RUN pip install --no-cache-dir -e ".[dev,bench]"

# Smoke test at build time so a broken image fails fast.
RUN pytest -q && foveance demo --budgets 1000,2500 --turns 18

# Default: run the offline benchmark + analysis + plots.
CMD ["bash", "-lc", "python bench/run_bench.py --backend mock --models mock --suite synthetic \
  --budgets 600,1200,2500 --tasks 4 --turns 30 --greedy-gap --ablations && \
  python bench/analyze.py && python bench/plots.py && cat bench/report.md"]
