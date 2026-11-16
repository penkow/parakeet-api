"""OpenAI-compatible /v1/audio/transcriptions on top of NVIDIA Parakeet-TDT-0.6B-v3 (NeMo).

- Parakeet-TDT predicts word timings natively (token durations), so word timestamps need no alignment model.
- The language is detected by the model (25 European languages); the `language` field is accepted but ignored.
- model: "whisper-1" (for OpenAI client compatibility), "parakeet" or the configured checkpoint name.
- `prompt` and `temperature` are accepted but ignored.
"""
import asyncio, gc, io, os, secrets, tempfile, threading, traceback
from contextlib import asynccontextmanager
from typing import Optional

import av
import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException

INDEX_HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), encoding="utf-8").read()

MODEL_NAME = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
# Full attention is fastest but memory use grows quadratically with length. Past this many seconds the model
# switches to local attention (NVIDIA: up to ~3 h of audio); word timestamps keep working.
LOCAL_ATTN_AFTER = float(os.environ.get("PARAKEET_LOCAL_ATTN_AFTER", "480"))
API_KEY = os.environ.get("API_KEY", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALIASES = {"whisper-1", "parakeet", "parakeet-tdt", "parakeet-tdt-0.6b-v3", MODEL_NAME.lower()}

_lock = threading.Lock()  # one GPU job at a time
_model = None


def get_model():
    global _model
    if _model is None:
        import nemo.collections.asr as nemo_asr  # heavy import, so done here and not at startup of the module

        print(f"loading {MODEL_NAME} on {DEVICE}", flush=True)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME, map_location=DEVICE)
        # NeMo's CUDA-graph RNNT decoder can crash with "illegal memory access" when other code allocates and
        # frees GPU memory in the same process (and it poisons the CUDA context). The plain decoder is only
        # slightly slower.
        from omegaconf import open_dict

        dec = model.cfg.decoding
        with open_dict(dec):
            dec.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(dec, verbose=False)
        model.eval()
        _model = model
        gc.collect()
    return _model


def decode_audio(data: bytes):
    """Any container/codec PyAV can read -> float32 mono 16 kHz."""
    chunks = []
    with av.open(io.BytesIO(data), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("no audio stream found")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(container.streams.audio[0]):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):  # flush
            chunks.append(f.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("audio is empty")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _group_words(segments, words):
    """Attach each word to the segment that contains its midpoint (both come from the same decoding pass)."""
    out = [{"start": g["start"], "end": g["end"], "text": g["text"], "words": []} for g in segments]
    if not out:
        return out
    i = 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        while i < len(out) - 1 and mid > out[i]["end"] and mid >= out[i + 1]["start"]:
            i += 1
        out[i]["words"].append(w)
    return out


def transcribe(audio):
    """Text, segment and word timestamps from one native pass. Returns a list of segments with their words."""
    with _lock:
        model = get_model()
        duration = len(audio) / 16000
        long_audio = duration > LOCAL_ATTN_AFTER
        if long_audio:
            model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, audio, 16000, subtype="PCM_16")
            with torch.inference_mode():
                hyp = model.transcribe([path], batch_size=1, timestamps=True, verbose=False)
            hyp = hyp[0] if isinstance(hyp, (list, tuple)) else hyp
            if isinstance(hyp, (list, tuple)):  # some RNNT versions return (best, all) tuples
                hyp = hyp[0]
        finally:
            os.unlink(path)
            if long_audio:
                model.change_attention_model(self_attention_model="rel_pos")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        stamps = hyp.timestamp or {}
        # `start`/`end` are seconds; fall back to encoder frames (80 ms each) if a version only gives offsets
        frame = float(model.cfg.preprocessor.window_stride) * 8

        def sec(d, key):
            return float(d[key]) if d.get(key) is not None else float(d[key + "_offset"]) * frame

        words = [{"word": w["word"], "start": round(sec(w, "start"), 3), "end": round(sec(w, "end"), 3)}
                 for w in stamps.get("word", [])]
        segs = [{"start": round(sec(g, "start"), 3), "end": round(sec(g, "end"), 3), "text": g["segment"]}
                for g in stamps.get("segment", [])]
        if not segs and hyp.text.strip():  # no segment info: one segment spanning the words
            segs = [{"start": words[0]["start"] if words else 0.0,
                     "end": words[-1]["end"] if words else duration, "text": hyp.text}]
        return _group_words(segs, words)


@asynccontextmanager
async def lifespan(app):
    def preload():
        with _lock:
            try:
                get_model()
            except Exception as e:
                print("preload failed:", e, flush=True)

    threading.Thread(target=preload, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(HTTPException)
async def openai_errors(request, exc):
    return JSONResponse({"error": {"message": exc.detail, "type": "invalid_request_error"}}, status_code=exc.status_code)


def auth(authorization: Optional[str] = Header(None)):
    if API_KEY:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(token, API_KEY):
            raise HTTPException(401, "Incorrect API key provided")


def ts(t, sep):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_NAME, "loaded": _model is not None}


@app.get("/v1/models")
def models(_=Depends(auth)):
    ids = list(dict.fromkeys(["whisper-1", "parakeet", MODEL_NAME]))
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "nvidia"} for i in ids]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request, _=Depends(auth)):
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(400, "'file' is required")
    model = (form.get("model") or "whisper-1").strip()
    if model.lower() not in ALIASES:
        raise HTTPException(400, f"model '{model}' is not served here; use 'parakeet' or '{MODEL_NAME}'")
    fmt = form.get("response_format") or "json"
    if fmt not in ("json", "text", "srt", "vtt", "verbose_json"):
        raise HTTPException(400, f"unsupported response_format '{fmt}'")

    try:
        audio = decode_audio(await upload.read())
    except Exception as e:
        raise HTTPException(400, f"could not decode audio: {e}")
    try:
        segs = await asyncio.to_thread(transcribe, audio)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"transcription failed: {e}")

    text = " ".join(s["text"].strip() for s in segs).strip()
    if fmt == "text":
        return PlainTextResponse(text + "\n")
    if fmt in ("srt", "vtt"):
        sep = "," if fmt == "srt" else "."
        out = ["WEBVTT\n"] if fmt == "vtt" else []
        for i, s in enumerate(segs, 1):
            if fmt == "srt":
                out.append(str(i))
            out.append(f"{ts(s['start'], sep)} --> {ts(s['end'], sep)}\n{s['text'].strip()}\n")
        return PlainTextResponse("\n".join(out), media_type="text/plain")
    if fmt == "json":
        return {"text": text}

    return {
        "task": "transcribe",
        "language": "auto",
        "duration": round(len(audio) / 16000, 3),
        "text": text,
        "segments": [
            {"id": i, "seek": 0, "start": s["start"], "end": s["end"], "text": s["text"],
             "tokens": [], "temperature": 0.0, "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 0.0}
            for i, s in enumerate(segs)
        ],
        "words": [w for s in segs for w in s["words"]],
    }
