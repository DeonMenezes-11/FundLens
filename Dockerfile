FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data/sources.json data/nav_snapshot.csv.gz ./data/
ENV FUNDLENS_DATA_DIR=/app/data
# downloads SEBI PDFs, builds chunks + embeddings + NAV database at image build time
RUN fundlens setup
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "fundlens.api:app", "--host", "0.0.0.0", "--port", "8000"]
