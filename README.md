Short-form video generator with TTS voice-over, background music, on-video text (static or karaoke), logo, and intro/outro. Works locally with ffmpeg and Python.

What it does per video
- Picks next video from `input_videos/`, scales/crops to target aspect, 30fps.
- Reads next quote from `quotes.txt` (UTF‑8 by default), overlays text.
- Generates TTS (OpenAI or gTTS fallback), mixes with background music with auto-ducking and 3s voice delay.
- Optionally bakes karaoke subtitles timed to the voice.
- Optionally overlays a logo and concatenates intro/outro.
- Saves unique-named result to `output_videos/`.

Requirements
- Python 3.10+
- ffmpeg in PATH (Windows builds: https://www.gyan.dev/ffmpeg/builds/)
- Install deps: `pip install -r requirements.txt`
- Optional: set `OPENAI_API_KEY` for OpenAI TTS; otherwise gTTS fallback is used.

Folders
- `input_videos/` — source videos (.mp4/.mov/.mkv…)
- `music/` — background music (.mp3/.wav/.m4a…)
- `quotes.txt` — one quote per line (UTF‑8 by default)
- `assets/` — your logo(s) (e.g., `assets/logo.png`)
- `clips/` — optional intro/outro clips
- `output_videos/` — results (auto-created, git-ignored)

Quick start
1) Install dependencies: `pip install -r requirements.txt`
2) (Optional) Set OpenAI key (PowerShell): `$env:OPENAI_API_KEY = "sk-..."`
3) Ensure ffmpeg is available: `ffmpeg -version`
4) Put videos in `input_videos/`, music in `music/`, write `quotes.txt`
5) Run to process ALL videos in input: `python make_videos.py`

Key options
- Aspect presets (default 9:16): `--aspect 9:16|1:1|16:9`
- Text mode: `--text_mode static|karaoke`
  - Karaoke color: `--karaoke_color lightblue|#RRGGBB`
  - Karaoke font: `--subtitle_fontname "Comic Sans MS"`
- Logo overlay: `--logo assets/logo.png --logo_pos bottom-right --logo_scale 160 --logo_margin 20`
- Intro/Outro: `--intro clips/intro.mp4 --outro clips/outro.mp4`
- Quotes encoding: `--quotes_encoding utf-8|cp1251`
- TTS model/voice: `--model gpt-4o-mini-tts --voice alloy` (OpenAI), fallback gTTS uses `--tts_lang`

Behavior and defaults
- Duration: 12s video; music starts at 0s; voice delayed 3s with gentle auto-ducking under voice.
- Text: auto-wrap, safe margins, dynamic font size so the whole quote stays visible.
- Static text shows from 3s; karaoke highlights words over voice duration.
- Processes all videos by default; `--count N` limits how many to render this run.

Examples
- All videos for Shorts/TikTok with karaoke and logo:
  `python make_videos.py --text_mode karaoke --karaoke_color lightblue --subtitle_fontname "Comic Sans MS" --logo assets/logo.png --logo_pos bottom-right --aspect 9:16`
- Horizontal (16:9) static text, with intro/outro:
  `python make_videos.py --aspect 16:9 --intro clips/intro.mp4 --outro clips/outro.mp4`

Publish to GitHub
1) Initialize Git and commit:
   - `git init`
   - `git add .`
   - `git commit -m "Initial release: TTS video generator"`
2) Create an empty repo on GitHub (no auto-README), copy its URL.
3) Add remote and push:
   - `git remote add origin https://github.com/USER/REPO.git`
   - `git branch -M main`
   - `git push -u origin main`

Notes
- This repo includes `.github/workflows/ci.yml` that installs deps and runs `python make_videos.py --help` on pushes.
- Do not commit secrets. Keep `OPENAI_API_KEY` in env vars locally or GitHub Actions secrets if you extend CI.
