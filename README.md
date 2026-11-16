# parakeet-api

A self-hosted speech-to-text server built on NVIDIA's
[Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). It has an **OpenAI-compatible
API** (`POST /v1/audio/transcriptions`), so any tool or SDK that talks to OpenAI's Whisper API can use it by
changing the base URL. It also comes with a small web page for drag-and-drop transcription.

You can run it with **Docker** or on **Kubernetes** (plain Deployment, or Knative to scale to zero when
it's idle).

- **Native word-level timestamps.** Parakeet is a Token-and-Duration Transducer (TDT): it predicts how long
  each token lasts, so word start/end times come straight out of the model in the same pass as the text.
  There is no forced-alignment step and no extra alignment model.
- 25 European languages, detected automatically, with punctuation and capitalisation
- Fast: an 11-minute file transcribes in about 17 seconds on an RTX 3060
- Output formats: `json`, `text`, `srt`, `vtt`, `verbose_json` (segments and words)
- Takes any audio or video file that FFmpeg/PyAV can read (mp3, wav, m4a, flac, ogg, mp4, mkv, webm, ...)
- Long files: switches to local attention automatically (NVIDIA: up to about 3 hours of audio)
- Optional API key

---

## Contents

- [Requirements](#requirements)
- [Quick start (Docker)](#quick-start-docker)
- [Docker in detail](#docker-in-detail)
- [Kubernetes](#kubernetes)
- [Using the API](#using-the-api)
- [Configuration](#configuration)
- [About the model](#about-the-model)
- [Troubleshooting](#troubleshooting)
- [How it works](#how-it-works)
- [Project layout](#project-layout)

---

## Requirements

- An **NVIDIA GPU** with about 4 GB of free VRAM (the model uses roughly 3.6 GB; long files with full
  attention use more, see `PARAKEET_LOCAL_ATTN_AFTER`)
- A recent NVIDIA driver on the host (CUDA 12.8 support, driver 570 or newer)
- Docker: the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Kubernetes: the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) or the
  [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/), so pods can request `nvidia.com/gpu`

You don't need CUDA installed on the host: the CUDA libraries come with the Python packages.

**Disk:** about 15 GB for the image or Python environment, plus about 2.5 GB for the model.

CPU-only machines were not tested. The server falls back to the CPU when it can't see a GPU, but expect it
to be slow.

---

## Quick start (Docker)

```bash
git clone https://github.com/penkow/parakeet-api.git
cd parakeet-api
cp .env.example .env
docker compose up -d --build
```

The first build downloads PyTorch, the CUDA libraries and NeMo (several GB), so it takes a while. When the
container is running, open **http://localhost:8000** and drop in a file, or call the API:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@meeting.mp3 \
  -F response_format=text
```

Watch the logs with `docker compose logs -f`. The model is downloaded on first start; `GET /health` shows
`"loaded": true` once it's ready. Importing NeMo is slow the first time (a few minutes on a slow disk).

> The Dockerfile has not been built by the author of this repo (there was no Docker on the development
> machine). The Kubernetes setup below is what was tested end to end, and it uses the same packages.

---

## Docker in detail

### Check the GPU works in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

If this fails, install or fix the NVIDIA Container Toolkit first (`sudo nvidia-ctk runtime configure --runtime=docker`
and restart Docker).

### docker compose

```bash
docker compose up -d --build      # build and start
docker compose logs -f            # follow logs
docker compose restart            # restart (e.g. after editing .env)
docker compose down               # stop (the downloaded model stays in the volume)
docker compose down -v            # stop and delete the model volume
```

The model is stored in the named volume `parakeet-data` (mounted at `/data`), so it's only downloaded once.

### Plain docker run

```bash
docker build -t parakeet-api .
docker run -d --name parakeet-api --gpus all -p 8000:8000 -v parakeet-data:/data parakeet-api
```

### Pick a specific GPU

On a machine with several GPUs, replace `count: 1` in `docker-compose.yml` with `device_ids: ["1"]`, or use
`--gpus '"device=1"'` with `docker run`.

### Push the image to a registry

```bash
docker build -t ghcr.io/<you>/parakeet-api:latest .
docker push ghcr.io/<you>/parakeet-api:latest
```

---

## Kubernetes

The Kubernetes setup **doesn't need an image registry**. It uses the stock `python:3.12-bookworm` image,
turns the files in `app/` into a ConfigMap, and builds the Python environment once on a PersistentVolume.
The first start takes 15-30 minutes while that installs (PyTorch, CUDA libraries and NeMo are several GB);
after that, pods start in seconds (plus the time to load the model).

Everything is wired up with Kustomize (built into `kubectl`). The `kustomization.yaml` in the repository
root is the entry point.

### 1. Install

```bash
git clone https://github.com/penkow/parakeet-api.git
cd parakeet-api
kubectl apply -k .
```

This creates, in namespace `parakeet`:

| Resource | Purpose |
|---|---|
| `PersistentVolumeClaim/parakeet-state` (30 Gi) | Python virtualenv + the downloaded model |
| `ConfigMap/parakeet-app-<hash>` | `server.py`, `index.html`, start and install scripts, `requirements.txt` |
| `Deployment/parakeet` | Init container builds the venv (skipped when it's up to date), then the server starts with 1 GPU |
| `Service/parakeet` | Port 80 → 8000 inside the cluster |

The PVC uses your cluster's default StorageClass. To pick another one, add `storageClassName` to
`k8s/pvc.yaml`.

### 2. Watch the first start

```bash
kubectl -n parakeet get pods -w
kubectl -n parakeet logs deploy/parakeet -c install -f    # venv build (first start only)
kubectl -n parakeet logs deploy/parakeet -f               # server + model download
```

### 3. Try it

```bash
kubectl -n parakeet port-forward svc/parakeet 8000:80
curl http://localhost:8000/health
```

Other pods in the cluster reach it at `http://parakeet.parakeet.svc.cluster.local`.

### Options (Kustomize components)

Uncomment the lines you need under `components:` in `kustomization.yaml` and run `kubectl apply -k .` again.
Keep them in the order listed there.

| Component | Use it when |
|---|---|
| `k8s/components/knative` | You run [Knative Serving](https://knative.dev/docs/serving/) and want it to **scale to zero** when idle (the GPU is free for other workloads). Replaces the Deployment and Service with a Knative Service. The first request after idle waits for the pod to start and load the model (NeMo's import alone can take minutes on a slow disk, so the first request may hit a gateway timeout; retry). |
| `k8s/components/nvidia-runtime` | **k3s or RKE2**, or any cluster where GPU pods need `runtimeClassName: nvidia`. With Knative, also enable the `kubernetes.podspec-runtimeclassname` feature flag in the `config-features` ConfigMap. |

Example for k3s with Knative:

```yaml
components:
- k8s/components/knative
- k8s/components/nvidia-runtime
```

With Knative the URL comes from the service name (`<name>.<your-domain>`). To change it, set `nameSuffix`
(uncomment it in `kustomization.yaml`): every object gets the suffix, so `nameSuffix: "-tts"` gives a service
called `parakeet-tts`. The same trick lets you run a second copy next to the first (also change `namespace`).

### Expose it outside the cluster

`k8s/ingress.yaml` is an example for ingress-nginx. Set the `host` (and `ingressClassName`), then uncomment
`- k8s/ingress.yaml` under `resources:`. Uploads can be big and transcription can take minutes, so keep the
body-size and timeout settings (or their equivalents on your ingress controller). With Knative, use your
Knative domain mapping instead.

If it's reachable from the internet, **set an API key** (next section).

### API key

```bash
kubectl -n parakeet create secret generic parakeet-api-key --from-literal=API_KEY="$(openssl rand -hex 24)"
kubectl -n parakeet rollout restart deploy/parakeet
```

Without that secret, the API is open. To read the key back:
`kubectl -n parakeet get secret parakeet-api-key -o jsonpath='{.data.API_KEY}' | base64 -d`

### Change settings or update

- **Settings** (`PARAKEET_MODEL`, `PARAKEET_LOCAL_ATTN_AFTER`): edit the `env` in `k8s/deployment.yaml`
  (or `k8s/components/knative/knative-service.yaml`), then `kubectl apply -k .`.
- **Code** (`app/`): edit and `kubectl apply -k .`. The ConfigMap gets a new hash, so the pod restarts on its own.
  With Knative this creates a new revision; the venv is reused.
- **Python packages** (`requirements.txt`): edit and `kubectl apply -k .`. The next start sees a different
  checksum and rebuilds the venv. With Knative, delete the older revision while the first build is still
  running, or two installs write to the same volume.

### Uninstall

```bash
kubectl delete -k .        # also deletes the PVC with the venv and the model
```

### Using your own image instead

If you prefer the Docker image from this repo (pushed to your registry) over building the venv on a volume,
change both containers in `k8s/deployment.yaml`:

- remove the `initContainers` block
- set `image: ghcr.io/<you>/parakeet-api:latest` and `command: ["/app/start.sh"]`
- remove the `PYTHON` env var and the `app` volume mount (the code is in the image)
- change the `state` mount to `/data` and the `HF_HOME` / `TORCH_HOME` values to match

---

## Using the API

The API follows OpenAI's [audio transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web page for drag-and-drop transcription |
| `GET` | `/health` | `{"status":"ok","device":"cuda","model":"nvidia/parakeet-tdt-0.6b-v3","loaded":true}` (no auth) |
| `GET` | `/v1/models` | Lists `whisper-1`, `parakeet` and the checkpoint name |
| `POST` | `/v1/audio/transcriptions` | Transcribe a file (multipart form) |

### Form fields for `/v1/audio/transcriptions`

| Field | Default | Notes |
|---|---|---|
| `file` | (required) | Audio or video file |
| `model` | `whisper-1` | `whisper-1` (accepted so OpenAI clients work unchanged), `parakeet` and the configured checkpoint name all mean the same model. Other names get a 400. |
| `response_format` | `json` | `json`, `text`, `srt`, `vtt`, `verbose_json` |
| `timestamp_granularities[]` | (none) | Accepted for compatibility. Word timestamps are native, so `verbose_json` always includes `words`. |
| `language`, `prompt`, `temperature` | (none) | Accepted for compatibility, ignored. The language is auto-detected. |

### curl

```bash
# Plain text
curl http://localhost:8000/v1/audio/transcriptions -F file=@talk.mp4 -F response_format=text

# Subtitles
curl http://localhost:8000/v1/audio/transcriptions -F file=@talk.mp4 -F response_format=srt -o talk.srt

# Segments + word timestamps
curl http://localhost:8000/v1/audio/transcriptions -F file=@talk.mp4 -F response_format=verbose_json

# With an API key
curl http://localhost:8000/v1/audio/transcriptions -H "Authorization: Bearer $API_KEY" -F file=@talk.mp4
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything-or-your-API_KEY")

with open("talk.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="parakeet",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )

print(result.text)
for w in result.words[:10]:
    print(w.word, w.start, w.end)
```

### OpenAI JavaScript SDK

```js
import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: process.env.API_KEY ?? "none" });
const result = await client.audio.transcriptions.create({
  model: "parakeet",
  file: fs.createReadStream("talk.mp3"),
});
console.log(result.text);
```

### `verbose_json` response

```json
{
  "task": "transcribe",
  "language": "auto",
  "duration": 11.0,
  "text": "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
  "segments": [{ "id": 0, "start": 0.24, "end": 10.56, "text": "And so, my fellow Americans, ask not ...", "...": "..." }],
  "words": [{ "word": "And", "start": 0.24, "end": 0.56 }, { "word": "so,", "start": 0.56, "end": 0.88 }, "..."]
}
```

Segments are sentences (split at `.`, `?`, `!`). Words keep their punctuation and capitalisation. Fields that
Whisper reports but Parakeet doesn't have (`avg_logprob`, `no_speech_prob`, ...) are zero.

### Long files

The whole file is decoded into memory and transcribed in one request. Up to `PARAKEET_LOCAL_ATTN_AFTER`
seconds (default 480) it uses full attention. Longer files switch to local attention automatically (NVIDIA
says up to about 3 hours), with word timestamps intact. For longer recordings, cut the audio into chunks
first (for example with `ffmpeg -f segment -segment_time 3600`) and add each chunk's start time to the
returned timestamps. Behind a reverse proxy, raise its upload size limit and timeouts.

---

## Configuration

Environment variables (in `.env` for Docker Compose, in the manifest `env` for Kubernetes):

| Variable | Default | Description |
|---|---|---|
| `PARAKEET_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | NeMo checkpoint to serve. Loaded at startup in the background. Other Parakeet TDT checkpoints should work too (not tested). |
| `PARAKEET_LOCAL_ATTN_AFTER` | `480` | Seconds of audio after which local attention is used to save memory. Lower it if long files run out of VRAM. |
| `API_KEY` | (empty) | When set, `/v1/*` requires `Authorization: Bearer <API_KEY>`. `/` and `/health` stay open. |
| `PORT` | `8000` | Docker Compose: port on the host. Inside the container the server always listens on 8000. |
| `HF_HOME`, `TORCH_HOME` | under `/data` (Docker), `/state` (Kubernetes) | Where the model is cached. |

---

## About the model

[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): 600M parameters,
FastConformer encoder with a Token-and-Duration Transducer decoder, released by NVIDIA under CC BY 4.0.

- **Languages:** Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German,
  Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak,
  Slovenian, Spanish, Swedish and Ukrainian. Other languages (Chinese, Japanese, Turkish, ...) are not
  supported; use a Whisper-based server for those.
- **Word timestamp accuracy:** timings come from encoder frames of 80 ms, so every value is a multiple of
  0.08 s. On one clean English clip, word starts differed from a WhisperX (wav2vec2) alignment by about 0.1 s
  on average (worst case 0.3 s), and word ends were about 0.12 s later on average, because a word before a
  pause tends to absorb part of the silence. That is fine for subtitles, search and click-to-seek; it is not
  frame-accurate, so don't use it for precise cuts or for measuring pauses. This was a comparison on a single
  short clip, not an accuracy benchmark.
- **Memory:** about 3.6 GB of VRAM once loaded.

---

## Troubleshooting

**`CUDA error: an illegal memory access was encountered`**
NeMo's CUDA-graph decoder can do this when other code shares the GPU process; this server turns that decoder
off. If you still see it, check that nothing else is modifying the CUDA context and restart the server (the
CUDA context is unusable after the error).

**`CUDA out of memory`**
Lower `PARAKEET_LOCAL_ATTN_AFTER` (for example to 120) so long files use local attention, and check nothing
else is using the GPU (`nvidia-smi`).

**`/health` shows `"device": "cpu"` on a GPU machine**
The container can't see the GPU. Docker: check the `nvidia-smi` test above. Kubernetes: check that the pod
requests `nvidia.com/gpu` and that the node advertises it (`kubectl describe node | grep nvidia.com/gpu`); on
k3s/RKE2 enable the `nvidia-runtime` component.

**Pod is stuck in `Pending`**
Usually no free GPU (`kubectl -n parakeet describe pod`) or the volume is still being provisioned. Only one
pod can hold a GPU at a time (unless your GPUs are shared); the Deployment uses the `Recreate` strategy for
this reason.

**The first request is slow or times out**
The model is downloaded (once) and loaded at startup, and importing NeMo takes a few minutes on a cold
disk. `/health` shows `"loaded": true` when it's ready. With Knative, a cold start after idle includes
starting the pod.

**`could not decode audio`**
The file has no audio track or a codec PyAV can't read. Try converting it:
`ffmpeg -i input.xyz -ac 1 -ar 16000 out.flac`.

**Hugging Face rate limits**
Set `HF_TOKEN` as an extra environment variable.

---

## How it works

- `app/server.py` is a FastAPI app. On startup it loads the model in the background.
- Each request is decoded with PyAV to 16 kHz mono, written to a temporary WAV file and passed to NeMo's
  `transcribe(..., timestamps=True)`, which returns the text plus word, segment and character timestamps.
  Words are then attached to the segment that contains them.
- A lock makes requests run one at a time on the GPU; concurrent requests queue up.
- `app/start.sh` starts uvicorn.
- `app/install.sh` (Kubernetes only) builds the virtualenv on the volume and records a checksum of
  `requirements.txt`, so it only reinstalls when that file changes.

---

## Project layout

```
.
├── app/
│   ├── server.py            # FastAPI server (OpenAI-compatible API)
│   ├── index.html           # web page served at /
│   ├── start.sh             # starts uvicorn
│   └── install.sh           # Kubernetes: builds the venv on the PVC
├── requirements.txt         # pinned Python packages
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── kustomization.yaml       # kubectl apply -k .
└── k8s/
    ├── namespace.yaml
    ├── pvc.yaml
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml         # optional
    └── components/
        ├── knative/         # scale to zero with Knative Serving
        └── nvidia-runtime/  # runtimeClassName: nvidia (k3s, RKE2)
```

## Credits

Built on NVIDIA's [NeMo](https://github.com/NVIDIA/NeMo) and the
[Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model (CC BY 4.0, © NVIDIA).
Check the licenses of the software and models you use before commercial use.
