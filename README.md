# AI Interview Analyzer

AI-powered mock interview platform that analyses a candidate's **answer content, speech, voice and body language** and produces an HR-style report.

> 🚧 Work in progress: built sprint by sprint.

## Laptop setup (PostgreSQL + pgAdmin)

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

### 3. Database
1. **pgAdmin** → right-click **Databases** → **Create → Database…** → name: `interview_ai` → Save.
2. Copy `.env.example` to `.env` and put your postgres password in `DATABASE_URL`.
3. Create all tables and default data:
   ```bash
   python -m backend.app.db.setup
   ```
   Expected output: `Database ready: 4 job roles, 18 questions.`

That's it. Refresh `interview_ai → Schemas → public → Tables` in pgAdmin to see them.

- The command is safe to run again; it only applies what is missing.
- The API also runs it automatically on startup.
- **Without Python:** open pgAdmin → `interview_ai` → **Query Tool** → open [`database/schema.sql`](database/schema.sql) → Run (F5).

### Database tables
| Table | Purpose |
|---|---|
| `job_roles` | Roles with description and required skills |
| `questions` | Question bank: ideal answer, keywords, time limit |
| `candidates` | Candidate profile and parsed resume analysis |
| `interviews` | One interview session: status, overall score, verdict, PDF path |
| `answers` | One recorded answer: media file, transcript, processing status |
| `answer_metrics` | All AI scores for an answer (speech, voice, vision, NLP, suggestions) |
| `alembic_version` | Migration version (managed by Alembic) |

### Changing the schema (developers)
1. Edit `backend/app/db/models.py`.
2. `alembic revision --autogenerate -m "describe change"`
3. `alembic upgrade head`
4. `python -m backend.app.db.export_sql` to refresh `database/schema.sql`.
