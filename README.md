# RedOak Clara Agent

Free voice agent for RedOak Global Services (medical billing / RCM).
One FastAPI app serves both the Twilio phone leg and the `/talk` test endpoint.

- **Voice pipeline:** Pipecat 1.16 + faster-whisper (`tiny`) + kokoro-onnx (`af_heart`)
- **LLM brain:** Pollinations free OpenAI-compatible API (no key). Set
  `LLM_BACKEND=ollama` to use a local Ollama model instead.
- **No torch** in this image: Pipecat 1.16 uses `kokoro_onnx`, not torch Kokoro.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + model status (keep-warm cron pings this) |
| POST | `/twilio-voice` | TwiML: streams the call to `wss://<PUBLIC_HOST>/ws` |
| WS | `/ws` | Twilio media stream → Pipecat pipeline |
| POST | `/talk` | wav upload → Clara reply (JSON with base64 wav, or `?format=wav` for raw wav) |

`/talk` accepts a multipart `file` field or a raw wav body, and returns
`user_transcript`, `agent_transcript`, `audio_base64`, `sample_rate`.
Transcripts are saved under `leads/` (the free CRM: `leads/<call_sid>.json`
for calls, `leads/talk-<timestamp>.json` for `/talk`).

## Environment variables

| Name | Default | Notes |
|---|---|---|
| `PORT` | `10000` | Set by Render |
| `PUBLIC_HOST` | `redoak-clara-agent.onrender.com` | Hostname only, no scheme/port. Set to the real hostname after first deploy, then redeploy |
| `LLM_BACKEND` | `openai-compatible` | `ollama` = local fallback |
| `POLLINATIONS_BASE_URL` | `https://text.pollinations.ai/openai` | Free, no key |
| `POLLINATIONS_MODEL` | `openai` | (= gpt-oss-20b) |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local fallback |
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Local fallback |
| `WHISPER_MODEL` | `tiny` | faster-whisper model |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice |
| `KOKORO_MODEL_PATH` | `./models/kokoro-v1.0.onnx` | Pre-downloaded in the Docker build |
| `KOKORO_VOICES_PATH` | `./models/voices-v1.0.bin` | Pre-downloaded in the Docker build |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` | (empty) | Only needed for the Twilio media stream serializer |
| `MAX_CALL_SECONDS` | `300` | Calls end after this |

## Twilio setup (after deploy)

1. In Twilio, set the phone number's voice webhook to
   `https://<PUBLIC_HOST>/twilio-voice`.
2. Call the number: Clara answers. The transcript is saved to `leads/<call_sid>.json`.

## Render free-plan caveats

- **512 MB RAM:** the talk pipeline + a Twilio call at the same time can OOM.
  One leg at a time is safe.
- **Sleep:** a free service sleeps after ~15 min idle; cold start is ~1-2 min.
  A cron pings `/health` every ~10 min to keep it warm.
- **Cold start:** the first `/talk` request loads the models (a few seconds).

## Local run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download kokoro models to ./models/ (see Dockerfile URLs)
uvicorn app:app --host 0.0.0.0 --port 7861
curl http://localhost:7861/health
```

## Files

- `app.py` — merged Twilio + `/talk` FastAPI app
- `redoak-agent-short-prompt.txt` — Clara's short system prompt
- `redoak-agent-knowledge-base.md` — RedOak facts for the agent
- `render.yaml` — Render Blueprint (Docker web service, free plan)
- `Dockerfile` — python:3.12-slim + ffmpeg, models pre-downloaded at build

Rebuilt 20 Nov 2026 from the handover spec (the previous workspace was lost
with the old VM). The prompt and knowledge base are reconstructions from the
handover rules and facts, not byte-identical copies of the lost originals.
