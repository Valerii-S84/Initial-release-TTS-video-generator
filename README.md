# 🎬 Auto Shorts/TikTok Video Generator  
**Text-to-speech voice-over + music ducking + on-screen text + (optional) karaoke subtitles, logo, intro/outro. Runs locally with Python + ffmpeg.**

---

## ✨ Що робить скрипт

Цей інструмент автоматично генерує короткі мотиваційні відео (12 секунд за замовчуванням) у форматі Reels / Shorts / TikTok.  
Для КОЖНОГО відео він:

1. 📹 Берe наступний файл із `input_videos/`, масштабує і кадрує під потрібний аспект (по дефолту 9:16), приводить до 30fps.
2. 📝 Читає наступну цитату з `quotes.txt` (одна цитата = один рядок).
3. 🔊 Генерує озвучку цитати (TTS).  
   - Використовує OpenAI TTS (якщо є `OPENAI_API_KEY`).
   - Якщо ключа немає — є fallback на gTTS.
4. 🎶 Міксує голос з фоновою музикою:
   - додає музичний трек з папки `music/`
   - робить авто-ducking (музика стає тихіше під час голосу)
   - голос входить із затримкою ~3 секунди
5. 💬 Додає текст у відео:
   - режим `static`: просто показує цитату внизу в боксі (з тінню і відступами)
   - режим `karaoke`: слова підсвічуються в ритм озвучки
6. 🌟 (опційно) Накладає логотип (водяний знак).
7. 🎬 (опційно) Додає готовий intro/outro кліп.
8. 💾 Зберігає результат у `output_videos/` з унікальним ім’ям.

Все відбувається повністю локально через `ffmpeg` і Python.

---

## 🗂 Структура проєкту

```text
.
├─ make_videos.py           # головний скрипт
├─ requirements.txt         # залежності Python
├─ input_videos/            # Відео-матеріал (вхід)
├─ music/                   # Музика для фону
├─ quotes.txt               # Цитати, одна на рядок (UTF-8)
├─ assets/
│   └─ logo.png             # (опційно) твій логотип / watermark
├─ clips/
│   ├─ intro.mp4            # (опційно) інтро
│   └─ outro.mp4            # (опційно) аутро
├─ output_videos/           # Готові результати (авто-створюється)
└─ .tmp_build/              # тимчасові файли (авто)

💻 Вимоги

Python 3.10+

Встановлений ffmpeg у PATH

Windows готові білди: https://www.gyan.dev/ffmpeg/builds/

pip install -r requirements.txt

(опційно) OPENAI_API_KEY для високоякісного TTS через OpenAI

Якщо API ключа немає → fallback gTTS (потрібен інтернет)

⚙️ Установка
1. Клон репозиторій
git clone <your-repo-url>.git
cd <your-repo-folder>

2. Встанови Python-залежності
pip install -r requirements.txt

3. Перевір ffmpeg
ffmpeg -version


Якщо команда не знаходиться — додай ffmpeg у PATH.

4. (Необов’язково) Укажи OpenAI ключ для голосу

PowerShell (Windows):

$env:OPENAI_API_KEY = "sk-..."


Без ключа скрипт автоматично спробує fallback через gTTS.

📥 Підготовка вхідних даних

Кинь відеофайли (.mp4, .mov, .mkv, …) у input_videos/

Кинь саундтреки (.mp3, .wav, .m4a, …) у music/

Створи quotes.txt, формат:

Твоє життя змінюється в той момент, коли ти вирішуєш не здаватися.
Сила приходить після болю. Терпи ще трохи.
...


Кожен рядок = окреме відео / окрема озвучка.

(опційно) Поклади логотип у assets/logo.png

(опційно) Поклади інтро/аутро в clips/intro.mp4, clips/outro.mp4

🚀 Швидкий старт

Згенерувати відео для ВСІХ клипів у input_videos/:

python make_videos.py


Це:

візьме твої відео

накладе текст (статичний режим)

зробить озвучку

зміксує з музикою

збереже результат у output_videos/

🛠 Ключові аргументи CLI

Ти можеш тонко контролювати результат через параметри командного рядка.

🎞 Аспект / розмір кадру
--aspect 9:16 | 1:1 | 16:9


9:16 → TikTok / Reels / Shorts (default, 720x1280)

1:1 → квадратний пост (1080x1080)

16:9 → горизонтальне відео YouTube (1920x1080)

Скрипт масштабує і кадрує так, щоб зберегти композицію (cover-fit + crop).

🗣 Озвучка (TTS)
--model gpt-4o-mini-tts
--voice alloy
--tts_lang uk


--model, --voice → використовуються для OpenAI TTS.

Якщо OPENAI_API_KEY немає, скрипт автоматично падає назад на gTTS.

--tts_lang (наприклад uk, en) використовується для gTTS.

🎙 Голос запускається не з нуля, а із затримкою (3 секунди за замовчуванням), щоб у відео спочатку трохи йшла музика, а потім вривається голос.

💬 Текст на екрані
--text_mode static|karaoke
--subtitle_fontname "Comic Sans MS"
--karaoke_color lightblue
--fontsize 56
--fontcolor "#ffffff"
--quotes_encoding utf-8


Режими:

static

внизу екрана з напівпрозорим чорним боксом

авто-розбиття на рядки

авто-зменшення шрифту, щоб цитата помістилась і не обрізалась

з’являється після ~3с (одночасно з голосом)

karaoke

генерується .ass субтитр, де кожне слово підсвічується синхронно з озвучкою

колір підсвітки контролюється --karaoke_color

шрифт керується --subtitle_fontname

В обох режимах скрипт намагається гарантувати, що текст не “вилазить” за межі відео.

🎧 Музика і ducking

Музика:

починається одразу з 0 секунди

залишається на фоні

під голосом автоматично притискається (ducking)

результат міксу — AAC 44.1 kHz stereo

Голос:

входить із затримкою 3s

нормалізується по гучності

виводиться чітко поверх фону

🖼 Логотип (watermark)
--logo assets/logo.png
--logo_pos bottom-right
--logo_scale 160
--logo_margin 20


Логотип масштабується до заданої ширини --logo_scale

Може стояти в одному з кутів:

top-left, top-right, bottom-left, bottom-right

Відступ від краю контролюється --logo_margin

🎬 Інтро/Аутро
--intro clips/intro.mp4
--outro clips/outro.mp4


Якщо ти передаєш інтро та/або аутро — фінальне відео збирається як:

[intro] + [основний ролик] + [outro]

Усі частини автоматично підганяються під той самий розмір кадру, fps і кодек.

🔢 Обмеження кількості
--count N


За замовчуванням скрипт обробляє ВСІ відео з input_videos/.

Якщо ти хочеш зробити, наприклад, тільки перші 3 — вкажи --count 3.

📌 Приклади запуску
1. Вертикальні відео для TikTok/Shorts з караоке-текстом і логотипом
python make_videos.py ^
  --aspect 9:16 ^
  --text_mode karaoke ^
  --karaoke_color lightblue ^
  --subtitle_fontname "Comic Sans MS" ^
  --logo assets/logo.png ^
  --logo_pos bottom-right

2. Горизонтальне відео (16:9), статичний текст, інтро+аутро
python make_videos.py ^
  --aspect 16:9 ^
  --text_mode static ^
  --intro clips/intro.mp4 ^
  --outro clips/outro.mp4

3. Використати fallback на gTTS українською (без OpenAI)
python make_videos.py ^
  --tts_lang uk

🔧 Важливі технічні деталі

Тривалість кожного основного ролика за замовчуванням: 12 секунд

FPS: 30

Кодування відео: H.264 (libx264, yuv420p) → оптимально для соцмереж

Аудіо: AAC stereo

Імена фінальних файлів у output_videos/ генеруються автоматично з timestamp + random suffix

Тимчасові файли лежать у .tmp_build/ і прибираються після успішної збірки

❗ Відомі обмеження

gTTS потребує інтернет-доступ, і голос змінити складно.

Якщо ffmpeg не в PATH — скрипт впаде з підказкою.

Якщо у шрифті немає кирилиці / діакритики, текст може показувати квадратики.
Рішення: передати --fontfile "C:/Windows/Fonts/arial.ttf" або інший ttf з кирилицею.

👤 Автор

Valerii Serputko
