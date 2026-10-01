# AI Interview Analyzer

An AI mock-interview platform for the IT department of K.E.S. Shroff College (B.Sc IT, B.Sc Data Science,
B.Sc AI, BCA). An AI interviewer **speaks each question**, listens to the spoken answer, checks it against a
model answer, **adapts the difficulty**, asks **counter questions**, and produces a detailed report. It also
interviews **IT-department teachers** on the subjects they teach.

## Features

| Area | What it does |
|---|---|
| Registration | Students: name, college, roll no, course (B.Sc IT / DS / AI / BCA), year (FY/SY/TY), resume. Teachers: course → year → subjects taught (multi-select), resume optional |
| Resume / ATS | 100-point explainable ATS score (pass ≥ 85%), every lost point with the exact mistake and fix; AI recruiter review; resume-based questions |
| Question bank | 90 syllabus subjects × 12 questions (2 per level) = 1,080 questions **with model answers**, from the KES Shroff syllabus (subjects from official KES exam time-tables, unit topics from Mumbai University syllabi) |
| Adaptive interview | 6 levels Basic → Professional. ≥ 85% correct → harder; 2 misses in a row → easier. 15 technical + 5-7 soft-skill/resume questions. Every candidate gets a different set |
| Counter questions | Follow-up questions built from the candidate's own answer. Teachers get 1-2 on **every** answer, never below Medium level |
| Voice | Interviewer voice (Microsoft neural Indian-English, edge-tts); answers transcribed with Whisper (faster-whisper) |
| Live feedback | "Correct answer!" / "This answer is not correct" with % score and what was missing |
| Review | At the end: "Do you want the answers?" Yes → teacher-style explanations, No → thank you |
| How you spoke | Fluency (pace, fillers, pauses), voice confidence (Praat prosody), tone (speech-emotion model), confident vs hesitant language |
| Body language | Optional live webcam: face in frame, eye contact (iris + head direction), smile, blinks, head and hand movement, facial emotion (MediaPipe + HSEmotion) |
| Report | Overall verdict, % for every question, knowledge by subject, level graph, delivery, body language, resume, experimental Big Five style indicator; downloadable HR-style PDF |
| Dashboard | Students and teachers, scores, ATS pass rate, CSV export |

## Machine-learning models (trained in this project)

| Model | Data | Result (held-out) | Used for |
|---|---|---|---|
| Speech emotion (`ml/models/voice_emotion.onnx`) | RAVDESS, 24 actors, speaker-independent test on 4 unseen actors | 54.2% on 8 emotions (chance 12.5%), 63.3% on calm/positive/nervous/tense | A tone hint only |
| Prosody calibration (`prosody_calibration.json`) | RAVDESS calm + neutral speech | natural ranges of jitter, shimmer, HNR | Voice-confidence thresholds |
| Big Five indicator (`big_five.onnx`) | Pennebaker & King Essays (2,467) | see `ml/models/big_five.json` | Report indicator only, never scored |

Retrain with `python -m ml.train_voice_emotion` / `python -m ml.train_big_five` (datasets are downloaded into
`ml/datasets/`, which is not committed). Answer grading uses an LLM (Groq, free) with an embedding + keyword
fallback when no key is configured.

## Laptop setup

### 1. Get the code
```bash
git clone https://github.com/prateekchau25-del/ai-interview-analyzer.git
cd ai-interview-analyzer
```
Already cloned? Just run `git pull`.

### 2. Python environment (Python 3.11 or 3.12)
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Also install FFmpeg: `winget install Gyan.FFmpeg`

### 3. Database (PostgreSQL + pgAdmin)
1. **pgAdmin** → right-click **Databases** → **Create → Database…** → name: `interview_ai` → Save.
2. Copy `.env.example` to `.env` and put your postgres password in `DATABASE_URL`.
3. Create all tables and load the question bank:
   ```bash
   python -m backend.app.db.setup
   ```
   Expected output: `Database ready: 4 job roles, 1409 questions.`

- Safe to run again; it only applies what is missing (also after every `git pull`).
- The API also runs it automatically on startup.
- **Without Python:** pgAdmin → `interview_ai` → **Query Tool** → open [`database/schema.sql`](database/schema.sql) → Run (F5).

### 4. Free LLM key (recommended)
Answer checking, explanations, counter questions and resume review are much better with an LLM.
1. Create a free key at https://console.groq.com/keys
2. In `.env`: `LLM_PROVIDER=groq` and `LLM_API_KEY=<your key>`

Without a key the app still works, using a stricter embedding + keyword checker.

### 5. Run
Two terminals (both with `.venv` activated):
```bash
uvicorn backend.app.main:app --port 8000
streamlit run frontend/app.py
```
Open http://localhost:8501. The first start downloads the Whisper model (~480 MB) once.

### Optional: run everything with Docker
If Docker Desktop is installed:
```bash
docker compose up --build
```
UI: http://localhost:8501 · API docs: http://localhost:8000/docs. This starts its own PostgreSQL
(host port 5433, so it does not clash with your pgAdmin server). Put `LLM_API_KEY=...` in `.env` first.

Note: on Windows/macOS Docker Desktop the browser webcam stream (WebRTC) often cannot reach a container.
For webcam body-language analysis run the UI without Docker (`streamlit run frontend/app.py`).

### 6. Tests
```bash
pytest -q
```
Tests use an in-memory database and never call the LLM.

## Project structure
```
backend/app/
  analyzers/   speech (Whisper), nlp (grading), lexical, prosody, voice_emotion, vision, personality, resume, tts, llm
  services/    interview_engine (adaptive logic), interview_service (orchestration), syllabus, report_pdf
  db/          models, Alembic migrations, seed, setup, export_sql
  data/        syllabus.json, question_bank/*.json, soft_skills.json, teacher_skills.json
  api/         FastAPI routes
frontend/      Streamlit app (register, interview, report, dashboard)
ml/            training scripts + trained ONNX models
database/      schema.sql for pgAdmin
tests/         unit, service, API and UI tests
```

### Changing the schema (developers)
1. Edit `backend/app/db/models.py`.
2. `alembic revision --autogenerate -m "describe change"`
3. `alembic upgrade head`
4. `python -m backend.app.db.export_sql` to refresh `database/schema.sql`.
