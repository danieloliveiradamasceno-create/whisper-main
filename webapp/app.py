import os
import re
import shutil
import sys
import tempfile
import threading
import uuid
import webbrowser

import torch
import yt_dlp
from flask import Flask, jsonify, request, send_from_directory

import whisper
from whisper.tokenizer import LANGUAGES
from whisper.utils import get_writer

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "transcricoes")
COOKIES_FILE = os.path.join(HERE, "cookies.txt")  # optional, takes priority over the browser
MODELS = ["tiny", "base", "small", "medium", "turbo", "large"]
BROWSERS = ["chrome", "edge", "firefox", "brave", "opera"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = None  # local app: no upload size limit

device = "cuda" if torch.cuda.is_available() else "cpu"
_models = {}
_lock = threading.Lock()  # one transcription at a time: avoids exhausting memory/VRAM

_jobs = {}
_current_job = None  # job being transcribed, read by the progress hook below


class _ProgressBar:
    """Stands in for tqdm inside whisper.transcribe to report progress per job."""

    def __init__(self, total=None, **kwargs):
        self.total = total or 1
        self.done = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n=1):
        self.done += n
        job = _current_job
        if job is not None:
            job["percent"] = min(100.0, 100.0 * self.done / self.total)


class _TqdmShim:
    tqdm = _ProgressBar


# whisper.transcribe is shadowed by the function of the same name in the package
sys.modules["whisper.transcribe"].tqdm = _TqdmShim


def get_model(name):
    if name not in _models:
        _models[name] = whisper.load_model(name, device=device)
    return _models[name]


def safe_name(name):
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return name[:120] or "transcricao"


def unique_name(base):
    name, i = base, 2
    while any(
        os.path.exists(os.path.join(OUTPUT_DIR, f"{name}.{ext}")) for ext in ("txt", "srt")
    ):
        name = f"{base} ({i})"
        i += 1
    return name


def download_audio(url, folder, browser, job):
    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            job["percent"] = 100.0 * d["downloaded_bytes"] / total if total else None

    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(folder, "video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    if os.path.exists(COOKIES_FILE):
        opts["cookiefile"] = COOKIES_FILE
    elif browser:
        opts["cookiesfrombrowser"] = (browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info), info.get("title")


def run_job(job, path, tmpdir, url, browser, display_name, model_name, language, task):
    global _current_job
    try:
        if url:
            job.update(stage="downloading", percent=None)
            try:
                path, title = download_audio(url, tmpdir, browser, job)
            except Exception as e:
                raise RuntimeError(
                    "Não foi possível baixar o vídeo: " + ANSI.sub("", str(e))
                ) from e
            display_name = title or "transcricao"

        job.update(stage="queued", percent=None)
        with _lock:
            job.update(stage="loading_model", percent=None)
            model = get_model(model_name)
            job.update(stage="transcribing", percent=0.0)
            _current_job = job
            try:
                result = model.transcribe(
                    path,
                    language=None if language == "auto" else language,
                    task=task,
                    fp16=(device == "cuda"),
                )
            finally:
                _current_job = None

        job.update(stage="saving", percent=None)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        name = unique_name(safe_name(display_name))
        for fmt in ("txt", "srt"):
            get_writer(fmt, OUTPUT_DIR)(result, name)

        job["result"] = {
            "text": result["text"].strip(),
            "language": result["language"],
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result["segments"]
            ],
            "name": name,
            "folder": OUTPUT_DIR,
            "files": [f"{name}.txt", f"{name}.srt"],
        }
        job.update(stage="done", percent=100.0)
    except Exception as e:  # e.g. ffmpeg missing, unreadable audio, out of memory
        app.logger.exception("Falha na transcrição")
        job.update(stage="error", error=str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.errorhandler(Exception)
def on_error(e):
    code = getattr(e, "code", 500)
    if code == 413:
        return jsonify(error="Arquivo grande demais."), 413
    app.logger.exception("Erro não tratado")
    return jsonify(error=f"{type(e).__name__}: {e}"), code if isinstance(code, int) else 500


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/info")
def info():
    return jsonify(
        device=device,
        models=MODELS,
        languages={code: name.title() for code, name in LANGUAGES.items()},
        output_dir=OUTPUT_DIR,
    )


@app.post("/transcribe")
def transcribe():
    file = request.files.get("audio")
    url = request.form.get("url", "").strip()
    if url:
        if not url.startswith(("http://", "https://")):
            return jsonify(error="Link inválido."), 400
    elif file is None or file.filename == "":
        return jsonify(error="Envie um arquivo de áudio ou informe um link."), 400

    model_name = request.form.get("model", "turbo")
    if model_name not in MODELS:
        return jsonify(error="Modelo inválido."), 400

    language = request.form.get("language", "auto")
    task = request.form.get("task", "transcribe")
    if task not in ("transcribe", "translate"):
        return jsonify(error="Tarefa inválida."), 400
    if language != "auto" and language not in LANGUAGES:
        return jsonify(error="Idioma inválido."), 400

    browser = request.form.get("browser", "")
    if browser and browser not in BROWSERS:
        return jsonify(error="Navegador inválido."), 400

    tmpdir = tempfile.mkdtemp()
    path = display_name = None
    if not url:
        stem, ext = os.path.splitext(file.filename)
        path = os.path.join(tmpdir, "input" + ext)
        file.save(path)
        display_name = stem

    job = {"stage": "queued", "percent": None, "error": None, "result": None}
    job_id = uuid.uuid4().hex
    _jobs[job_id] = job
    threading.Thread(
        target=run_job,
        args=(job, path, tmpdir, url, browser or None, display_name, model_name, language, task),
        daemon=True,
    ).start()
    return jsonify(job_id=job_id)


@app.get("/progress/<job_id>")
def progress(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return jsonify(error="Tarefa não encontrada."), 404
    return jsonify(
        stage=job["stage"], percent=job["percent"], error=job["error"], result=job["result"]
    )


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print(f"Whisper rodando em {url} (dispositivo: {device})")
    print(f"Transcrições salvas em: {OUTPUT_DIR}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5000, threaded=True)
