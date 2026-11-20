"""RedOak Clara — free voice agent (merged Twilio + /talk app).

One FastAPI app for Render (Docker, free plan):
  - GET  /health        liveness + model status (also used by the keep-warm cron)
  - POST /twilio-voice  TwiML: streams the call to wss://<PUBLIC_HOST>/ws
  - WS   /ws            Twilio media stream -> Pipecat pipeline (STT + LLM + TTS)
  - POST /talk          wav upload -> Clara reply (JSON with base64 wav, or ?format=wav)

Stack: Pipecat 1.16 + faster-whisper (tiny) + kokoro-onnx (af_heart).
LLM brain: Pollinations free OpenAI-compatible API (no key). Set
LLM_BACKEND=ollama to use a local Ollama model instead.

Env vars (see render.yaml / README.md):
  PORT, PUBLIC_HOST (hostname only, no scheme/port), LLM_BACKEND,
  POLLINATIONS_BASE_URL, POLLINATIONS_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL,
  WHISPER_MODEL, KOKORO_VOICE, KOKORO_MODEL_PATH, KOKORO_VOICES_PATH,
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, LEADS_DIR, MAX_CALL_SECONDS.

Rebuilt 20 Nov 2026 from the handover spec (previous workspace was lost
with the old VM). Behaviour matches the old bot.py + talk_server.py:
same endpoints, same models, same prompt/KB files.
"""

import asyncio
import base64
import io
import json
import logging
import os
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_HOST = os.environ.get("PUBLIC_HOST", "redoak-clara-agent.onrender.com")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "openai-compatible").lower()
POLLINATIONS_BASE_URL = os.environ.get(
    "POLLINATIONS_BASE_URL", "https://text.pollinations.ai/openai"
)
POLLINATIONS_MODEL = os.environ.get("POLLINATIONS_MODEL", "openai")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
KOKORO_MODEL_PATH = os.environ.get(
    "KOKORO_MODEL_PATH", str(BASE_DIR / "models" / "kokoro-v1.0.onnx")
)
KOKORO_VOICES_PATH = os.environ.get(
    "KOKORO_VOICES_PATH", str(BASE_DIR / "models" / "voices-v1.0.bin")
)
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
LEADS_DIR = Path(os.environ.get("LEADS_DIR", str(BASE_DIR / "leads")))
MAX_CALL_SECONDS = int(os.environ.get("MAX_CALL_SECONDS", "300"))

# /talk limits (free tier: a fraction of one CPU — bound every stage so a
# single long call cannot choke the server or queue behind it forever).
TALK_LLM_TIMEOUT_S = 60         # one Pollinations LLM call
TALK_PIPELINE_TIMEOUT_S = 150   # whole STT -> LLM -> TTS pipeline
TALK_LOCK_TIMEOUT_S = 10        # wait this long for a busy server, then 503
TALK_MAX_AUDIO_SECONDS = 120    # server-side cap on uploaded audio length

LEADS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("redoak-clara")

# ---------------------------------------------------------------------------
# Prompt + knowledge base (loaded once at startup)
# ---------------------------------------------------------------------------


def _read_text(name: str) -> str:
    path = BASE_DIR / name
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        log.warning("missing %s, continuing with empty text", name)
        return ""


SYSTEM_PROMPT = _read_text("redoak-agent-short-prompt.txt")
KNOWLEDGE_BASE = _read_text("redoak-agent-knowledge-base.md")
SYSTEM_WITH_KB = (
    SYSTEM_PROMPT + "\n\nKNOWLEDGE BASE (use for facts, never read out verbatim):\n"
    + KNOWLEDGE_BASE
    if KNOWLEDGE_BASE
    else SYSTEM_PROMPT
)

FIRST_MESSAGE = (
    "Hello, this is Clara from RedOak Global Services. Do you have 30 seconds? "
    "We help small practices with medical billing and AR follow-up."
)

# ---------------------------------------------------------------------------
# Shared /talk models (lazy-loaded, kept warm after first use)
# ---------------------------------------------------------------------------

_talk_lock = asyncio.Lock()  # one /talk request at a time (free-tier RAM)
_stt_model = None  # faster_whisper.WhisperModel
_tts_engine = None  # kokoro_onnx.Kokoro


def get_stt_model():
    global _stt_model
    if _stt_model is None:
        from faster_whisper import WhisperModel

        log.info("loading whisper model %s (cpu, int8)", WHISPER_MODEL)
        _stt_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _stt_model


def get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        from kokoro_onnx import Kokoro

        log.info("loading kokoro model %s", KOKORO_MODEL_PATH)
        _tts_engine = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
    return _tts_engine


def llm_client_kwargs():
    """Return (base_url, model) for the configured LLM backend."""
    if LLM_BACKEND == "ollama":
        return OLLAMA_BASE_URL, OLLAMA_MODEL
    return POLLINATIONS_BASE_URL, POLLINATIONS_MODEL


async def chat_reply(user_text: str, system: str = SYSTEM_WITH_KB) -> str:
    """One LLM turn via the OpenAI-compatible API (Pollinations or Ollama)."""
    from openai import AsyncOpenAI

    base_url, model = llm_client_kwargs()
    client = AsyncOpenAI(
        api_key="not-needed", base_url=base_url, timeout=TALK_LLM_TIMEOUT_S
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    return (resp.choices[0].message.content or "").strip()


def wav_bytes_to_float16k(audio_bytes: bytes) -> np.ndarray:
    """Decode wav bytes -> float32 mono 16kHz numpy array for faster-whisper.

    Done with stdlib `wave` + numpy on purpose: faster-whisper's file-path
    decode calls av.open(..., metadata_errors="ignore"), which av 19
    rejects (faster-whisper 1.2.1 pins no av version, pip picks av 19).
    Passing an ndarray skips faster-whisper's av decode entirely.
    Raises ValueError if the bytes are not a decodable wav.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"could not decode wav: {exc}") from exc
    if not frames:
        raise ValueError("empty wav")
    if sampwidth == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav sample width: {sampwidth}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if framerate != 16000:
        # Linear resample to 16kHz (whisper expects 16kHz ndarray input).
        # Done with output-sized float32 index arrays on purpose: the old
        # np.interp(np.linspace(...), np.arange(len)) built two float64
        # arrays as large as the INPUT (115MB for 7 min at 48kHz), which
        # could OOM the 1GB container on top of the loaded models. This
        # version only allocates output-sized arrays (positions stay
        # float64 — float32 loses index precision on long inputs).
        n_out = int(round(len(audio) * 16000 / framerate))
        pos = np.linspace(0, len(audio) - 1, n_out)
        i0 = np.floor(pos).astype(np.int32)
        frac = pos - i0
        i1 = np.minimum(i0 + 1, len(audio) - 1)
        audio = (
            audio[i0] * (1.0 - frac) + audio[i1] * frac
        ).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def wav_duration_seconds(audio_bytes: bytes) -> float | None:
    """Read a wav's duration from its header only — no sample decode, so a
    huge upload can be rejected before it ever allocates big arrays.
    Returns None when the bytes are not a readable wav header."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            if wf.getframerate() <= 0:
                return None
            return wf.getnframes() / wf.getframerate()
    except (wave.Error, EOFError):
        return None


def pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """float32 mono audio in [-1, 1] -> 16-bit PCM WAV bytes."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def save_lead(filename: str, payload: dict) -> str:
    path = LEADS_DIR / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RedOak Clara Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "redoak-clara-agent",
        "llm_backend": LLM_BACKEND,
        "public_host": PUBLIC_HOST,
        "stt_loaded": _stt_model is not None,
        "tts_loaded": _tts_engine is not None,
    }


@app.get("/")
async def index():
    return {
        "service": "RedOak Clara Agent",
        "endpoints": ["GET /health", "POST /twilio-voice", "WS /ws", "POST /talk"],
    }


@app.get("/test")
async def live_test_page():
    """Serve the Clara Live Test page (same origin as /talk, so the browser
    microphone works on this top-level page)."""
    return FileResponse(BASE_DIR / "static" / "test.html", media_type="text/html")


# ---------------------------------------------------------------------------
# Twilio voice webhook -> TwiML
# ---------------------------------------------------------------------------


@app.post("/twilio-voice")
async def twilio_voice():
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        "  <Connect>\n"
        f'    <Stream url="wss://{PUBLIC_HOST}/ws" />\n'
        "  </Connect>\n"
        "</Response>\n"
    )
    return PlainTextResponse(twiml, media_type="application/xml")


# ---------------------------------------------------------------------------
# Twilio media stream -> Pipecat pipeline
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def twilio_ws(websocket: WebSocket):
    await websocket.accept()
    log.info("twilio websocket connected")

    # Twilio sends a "start" event first with stream/call SIDs.
    stream_sid = None
    call_sid = None
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("event") == "start":
                start = data.get("start", {})
                stream_sid = start.get("streamSid")
                call_sid = start.get("callSid")
                log.info("twilio start: stream=%s call=%s", stream_sid, call_sid)
                break
            # Ignore "connected"/"mark" events before start.
    except WebSocketDisconnect:
        log.info("twilio websocket disconnected before start")
        return
    except Exception:
        log.exception("error waiting for twilio start event")
        return

    if not stream_sid or not call_sid:
        log.error("twilio start event missing stream/call SID")
        await websocket.close()
        return

    # Pipecat imports are done here so /health and /talk keep working even
    # if the realtime stack has an issue. API verified against pipecat 1.16:
    # transport lives in transports.websocket.fastapi, VAD is a pipeline
    # processor (VADProcessor), context is LLMContext + LLMContextAggregatorPair,
    # and PipelineWorker replaces the deprecated PipelineTask.
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
    )
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.services.kokoro.tts import KokoroTTSService
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    base_url, model = llm_client_kwargs()

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=TwilioFrameSerializer(
                stream_sid,
                call_sid,
                TWILIO_ACCOUNT_SID or None,
                TWILIO_AUTH_TOKEN or None,
            ),
        ),
    )

    stt = WhisperSTTService(model=WHISPER_MODEL)
    llm = OpenAILLMService(api_key="not-needed", base_url=base_url, model=model)
    # Reuse the pre-downloaded Kokoro models (Dockerfile); the service would
    # otherwise download the same files again on first use.
    tts = KokoroTTSService(
        voice_id=KOKORO_VOICE,
        model_path=KOKORO_MODEL_PATH,
        voices_path=KOKORO_VOICES_PATH,
    )

    context = LLMContext(
        [
            {"role": "system", "content": SYSTEM_WITH_KB},
        ]
    )
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=MAX_CALL_SECONDS,
    )

    # Greet first: the system prompt tells Clara to open with FIRST_MESSAGE.
    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        await worker.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False)
    try:
        await asyncio.wait_for(runner.run(worker), timeout=MAX_CALL_SECONDS)
    except asyncio.TimeoutError:
        log.info("call %s reached max duration (%ss), ending", call_sid, MAX_CALL_SECONDS)
        await worker.cancel()
    except WebSocketDisconnect:
        log.info("twilio websocket disconnected during call %s", call_sid)
    except Exception:
        log.exception("error in pipecat pipeline for call %s", call_sid)
    finally:
        try:
            messages = context.get_messages()
        except Exception:
            messages = []
        save_lead(
            f"{call_sid}.json",
            {
                "call_sid": call_sid,
                "stream_sid": stream_sid,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "messages": messages,
            },
        )
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# /talk — wav upload -> Clara reply (used by the Clara Live Test page)
# ---------------------------------------------------------------------------


class _ClientGone(Exception):
    """The /talk client disconnected mid-call; abort instead of crunching on."""


async def _run_talk_pipeline(request: Request, audio_np: np.ndarray):
    """STT -> LLM -> TTS. Every blocking step runs in a thread, so the event
    loop stays free and /health answers while a call is in progress.
    Returns (user_text, agent_text, wav_bytes, sample_rate).
    """
    # 1. Transcribe (fresh context per call — no history carried over).
    stt_model = await asyncio.to_thread(get_stt_model)
    segments, _info = await asyncio.to_thread(
        stt_model.transcribe, audio_np, beam_size=1
    )
    user_text = " ".join(seg.text for seg in segments).strip()
    if not user_text:
        user_text = "(no speech detected)"

    if await request.is_disconnected():
        raise _ClientGone("client disconnected after STT")

    # 2. LLM reply.
    agent_text = await chat_reply(user_text)
    if not agent_text:
        agent_text = (
            "Sorry, I didn't catch that. Could you tell me a little about "
            "your practice's billing setup?"
        )

    if await request.is_disconnected():
        raise _ClientGone("client disconnected after LLM")

    # 3. Text to speech.
    tts_engine = await asyncio.to_thread(get_tts_engine)
    audio, sample_rate = await asyncio.to_thread(
        tts_engine.create, agent_text, voice=KOKORO_VOICE, speed=1.0, lang="en-us"
    )
    wav_bytes = await asyncio.to_thread(
        pcm_to_wav_bytes, np.asarray(audio, dtype=np.float32), int(sample_rate)
    )
    return user_text, agent_text, wav_bytes, int(sample_rate)


@app.post("/talk")
async def talk(request: Request, file: UploadFile | None = File(default=None)):
    """Accept a wav upload (multipart `file` field, or raw wav body).

    Returns JSON by default:
      {"user_transcript": ..., "agent_transcript": ...,
       "audio_base64": ..., "sample_rate": 24000, "format": "wav"}
    With ?format=wav the reply wav bytes are returned directly, with the
    transcripts in X-User-Transcript / X-Agent-Transcript headers.

    Guardrails (free tier, one slow CPU): audio is capped at 120s, the
    pipeline fails fast after 150s (504), and a busy server answers 503
    instead of queueing forever. Nothing here blocks the event loop, so
    /health stays fast while a call runs.
    """
    audio_bytes: bytes | None = None
    if file is not None:
        audio_bytes = await file.read()
    else:
        body = await request.body()
        if body:
            audio_bytes = body

    if not audio_bytes:
        return JSONResponse({"error": "no audio uploaded"}, status_code=400)

    want_wav = request.query_params.get("format", "").lower() == "wav"

    # Reject over-long audio from the WAV header BEFORE decoding: decoding a
    # multi-minute recording allocates hundreds of MB of numpy temporaries,
    # which OOM-killed the 1GB container (20 Nov: /talk 502'd, container
    # restarted with models unloaded).
    header_s = wav_duration_seconds(audio_bytes)
    if header_s is not None and header_s > TALK_MAX_AUDIO_SECONDS:
        return JSONResponse(
            {"error": f"audio too long ({header_s:.0f}s, max {TALK_MAX_AUDIO_SECONDS}s)"},
            status_code=400,
        )

    # Decode off the event loop: for a long recording the numpy resample
    # takes seconds, which used to freeze /health mid-call.
    try:
        audio_np = await asyncio.to_thread(wav_bytes_to_float16k, audio_bytes)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    duration_s = len(audio_np) / 16000
    if duration_s > TALK_MAX_AUDIO_SECONDS:
        return JSONResponse(
            {"error": f"audio too long ({duration_s:.0f}s, max {TALK_MAX_AUDIO_SECONDS}s)"},
            status_code=400,
        )

    # One request at a time, but never queue forever.
    try:
        await asyncio.wait_for(_talk_lock.acquire(), timeout=TALK_LOCK_TIMEOUT_S)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": "server busy, please try again in a moment"}, status_code=503
        )

    try:
        try:
            user_text, agent_text, wav_bytes, sample_rate = await asyncio.wait_for(
                _run_talk_pipeline(request, audio_np),
                timeout=TALK_PIPELINE_TIMEOUT_S,
            )
        except _ClientGone:
            log.info("/talk: client disconnected mid-call, aborting")
            return JSONResponse({"error": "client disconnected"}, status_code=499)
        except asyncio.TimeoutError:
            log.warning(
                "/talk: pipeline exceeded %ss, failing fast",
                TALK_PIPELINE_TIMEOUT_S,
            )
            return JSONResponse(
                {"error": "Clara took too long to reply, please try again"},
                status_code=504,
            )
    finally:
        _talk_lock.release()

    # 4. Save the exchange (CRM = leads/*.json), file I/O off the loop.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    await asyncio.to_thread(
        save_lead,
        f"talk-{stamp}.json",
        {
            "source": "talk",
            "at": datetime.now(timezone.utc).isoformat(),
            "user_transcript": user_text,
            "agent_transcript": agent_text,
        },
    )

    if want_wav:
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-User-Transcript": user_text,
                "X-Agent-Transcript": agent_text,
            },
        )
    audio_b64 = await asyncio.to_thread(base64.b64encode, wav_bytes)
    return JSONResponse(
        {
            "user_transcript": user_text,
            "agent_transcript": agent_text,
            "audio_base64": audio_b64.decode("ascii"),
            "sample_rate": int(sample_rate),
            "format": "wav",
        }
    )
