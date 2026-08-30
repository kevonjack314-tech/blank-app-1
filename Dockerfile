# Self-hosting image. Streamlit Community Cloud does not use this - it reads
# requirements.txt directly - but any container host will.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nflproj/ ./nflproj/
COPY scripts/ ./scripts/
COPY data/coaching_2026.yaml ./data/coaching_2026.yaml
# Hand-entered beat reporting. The app runs without it, but shipping it empty
# loses whatever notes were written.
COPY data/news_2026.yaml ./data/news_2026.yaml
COPY streamlit_app.py METHOD.md README.md ./

# Warm the nflverse cache at build time so the first visitor does not wait for
# a 140 MB download. Comment this out to fetch lazily on first run instead.
RUN python -c "from nflproj import data; data.sync_all(projection_season=2026)" || true

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
