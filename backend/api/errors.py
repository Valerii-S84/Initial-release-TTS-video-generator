from __future__ import annotations

# Error codes
BAD_OFFSET = "BAD_OFFSET"
NOT_FOUND = "NOT_FOUND"
FORBIDDEN = "FORBIDDEN"
VALIDATION_ERROR = "VALIDATION_ERROR"

# Messages (UA)
MSG_VIDEO_NOT_FOUND = "Відео не знайдено"
MSG_UPLOAD_SESSION_NOT_FOUND = "Сесію завантаження не знайдено"
MSG_UPLOAD_FORBIDDEN = "Немає прав на цю сесію"
MSG_BAD_OFFSET = "Некоректний offset"
MSG_INVALID_FILE_SIZE = "Некоректний розмір файлу"
MSG_FFPROBE_FAILED = "Перевірка ffprobe не пройдена"
MSG_UPLOAD_INCOMPLETE = "Завантаження не завершено"
MSG_CHUNK_TOO_LARGE = "Розмір chunk перевищує дозволений ліміт"

def err(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}

