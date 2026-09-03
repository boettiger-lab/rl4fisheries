# ---- Base image ----
FROM pytorch/pytorch:2.x-cuda12.x-cudnn8-runtime AS base

# Avoid .pyc files, buffer stdout/stderr immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Install dependencies (layer caching) ----
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip
COPY . .
RUN pip install .

# ---- Entrypoint ----
RUN chmod +x /welcome.sh
ENTRYPOINT ["welcome.sh"]
CMD ["scripts/trian_rl_algos.sh"]