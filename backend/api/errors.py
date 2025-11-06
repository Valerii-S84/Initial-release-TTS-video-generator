from __future__ import annotations

# Standard error codes
BAD_OFFSET = "BAD_OFFSET"
NOT_FOUND = "NOT_FOUND"
FORBIDDEN = "FORBIDDEN"
VALIDATION_ERROR = "VALIDATION_ERROR"
BROKER_UNAVAILABLE = "BROKER_UNAVAILABLE"
SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
QUOTE_TOO_LONG = "QUOTE_TOO_LONG"
UNSUPPORTED_VIDEO_FORMAT = "UNSUPPORTED_VIDEO_FORMAT"
CORRUPTED_FILE = "CORRUPTED_FILE"
RATE_LIMIT = "RATE_LIMIT"

# Default UA messages for known codes (used if message not provided)
DEFAULT_MESSAGES = {
    BAD_OFFSET: "Невірний offset фрагмента завантаження",
    NOT_FOUND: "Об'єкт не знайдено",
    FORBIDDEN: "Доступ заборонено",
    VALIDATION_ERROR: "Помилка валідації вхідних даних",
    BROKER_UNAVAILABLE: "Черга задач тимчасово недоступна",
    SERVICE_UNAVAILABLE: "Сервіс тимчасово недоступний",
    QUOTE_TOO_LONG: "Цитата занадто довга",
    UNSUPPORTED_VIDEO_FORMAT: "Непідтримуваний формат відео",
    CORRUPTED_FILE: "Файл пошкоджено або не читається",
    RATE_LIMIT: "Перевищено ліміт запитів",
}


def err(code: str, message: str | None = None, hint: str | None = None) -> dict:
    payload = {"code": code, "message": message or DEFAULT_MESSAGES.get(code, "Помилка")}
    if hint:
        payload["hint"] = hint
    return {"error": payload}

# Backward-compatible named messages (kept for imports)
MSG_VIDEO_NOT_FOUND = DEFAULT_MESSAGES[NOT_FOUND]
MSG_UPLOAD_SESSION_NOT_FOUND = DEFAULT_MESSAGES[NOT_FOUND]
MSG_UPLOAD_FORBIDDEN = DEFAULT_MESSAGES[FORBIDDEN]
MSG_BAD_OFFSET = DEFAULT_MESSAGES[BAD_OFFSET]
MSG_INVALID_FILE_SIZE = DEFAULT_MESSAGES[VALIDATION_ERROR]
MSG_FFPROBE_FAILED = "Не вдалося отримати метадані відео (ffprobe)"
MSG_UPLOAD_INCOMPLETE = "Завантаження файлу не завершено"
MSG_CHUNK_TOO_LARGE = "Розмір chunk перевищує дозволений ліміт"
