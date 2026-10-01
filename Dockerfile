# One image for both the API and the Streamlit UI (CPU only).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models-cache/huggingface

# ffmpeg: audio extraction; libgl/libglib: OpenCV + MediaPipe; libsndfile: soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY ml ./ml
COPY database ./database
COPY alembic.ini pyproject.toml ./

EXPOSE 8000 8501
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
