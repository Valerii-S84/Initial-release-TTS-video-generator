# Video Generator API

FastAPI‑бекенд для обробки відео: завантаження (пряме або chunk), TTS, музика, генерація кліпів, черги Celery.

## Огляд
- API: backend/api/v1/{videos, jobs, admin, auth}
- Зберігання: direct/chunk upload з перевіркою ffprobe
- Генерація: video_generator → Celery task → video_engine
- TTS: ElevenLabs SDK
- Дані: SQLAlchemy, Alembic; Redis як кеш/черги (з in‑memory fallback)
- Спостережуваність: Prometheus, OpenTelemetry, Sentry

## Швидкий старт
- Локально
  - python -m venv .venv && .venv\Scripts\pip install -r requirements-dev.txt
  - set DATABASE_URL=sqlite:///backend/users.db
  - set JWT_SECRET_KEY=dev-secret
  - uvicorn backend.main:app --reload
- Docker
  - docker build -t videogen-api:dev .
  - docker run -p 8000:8000 --env-file .env videogen-api:dev

## Конфігурація (ENV)
- Обов’язкові: APP_ENV (dev|staging|prod), JWT_SECRET_KEY, DATABASE_URL
- Рекомендовані: REDIS_URL, CELERY_BROKER_URL, CELERY_RESULT_BACKEND, ELEVENLABS_API_KEY
- Сховище: STORAGE_INPUT, STORAGE_OUTPUT, STORAGE_TMP; MUSIC_DIR, MUSIC_PREVIEWS
- Ліміти: MAX_UPLOAD_SIZE_MB, MAX_CHUNK_SIZE_MB (деф. 8), ENFORCE_UPLOAD_TOKEN

## Безпека
- У prod заборонено порожній/дефолтний JWT_SECRET_KEY (fail‑fast)
- Рекомендації: secrets manager, валідні REDIS/CELERY URL, token для chunk‑сесій

## Політика помилок
- 422: схемні/структурні помилки (Pydantic)
- 400: бізнес‑валідації (size ≤ 0, перевищення ліміту chunk, incomplete, ffprobe)
- Коди/повідомлення — backend/api/errors.py

## Огляд API (v1/videos)
- GET /api/v1/videos — список; page, size(≤100), sort, include_deleted, q
- POST /api/v1/videos/upload — direct upload (multipart/form-data file)
- POST /api/v1/videos/upload/init — {filename, size} → {upload_id, tmp_path, token}
- PATCH /api/v1/videos/upload/chunk — upload_id, offset, optional token
- POST /api/v1/videos/upload/finish — завершення сесії
- POST /api/v1/videos/generate — GeneratePayload (м’яка валідація), бізнес‑перевірки в build_generate_cfg
- DELETE /api/v1/videos/{id}, POST /api/v1/videos/{id}/restore

## Тестування і покриття
- Запуск: python -m pytest
- Поріг покриття: мінімум 93%
- Звіт: htmlcov/index.html
- Маркери: unit, integration, isolated

## CI/CD
- GitHub Actions: лінтинг, типи, безпека, тести з coverage
- PR блокується при покритті < 93%
- HTML‑coverage публікується як артефакт

## Дорожня карта
- Celery beat для регулярної очистки тимчасових файлів/сесій
- Додаткові fail‑fast для ELEVENLABS/REDIS/CELERY у prod
- S3/GCS для INPUT/OUTPUT, підписані URL

© 2025 Valerii Serputko