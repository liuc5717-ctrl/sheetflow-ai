import os
import time
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field, field_validator

load_dotenv()

app = FastAPI(title="SheetFlow AI API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
SHEETFLOW_MODEL = os.getenv("SHEETFLOW_MODEL", "qwen/qwen3.8-27b:free").strip()
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "12"))
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "2000"))
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "45"))

MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "SHEETFLOW_MODEL_FALLBACKS",
        "google/gemma-3-4b-it:free,meta-llama/llama-3.3-70b-instruct:free,openai/gpt-oss-20b:free",
    ).split(",")
    if m.strip()
]

MODEL_CANDIDATES = [SHEETFLOW_MODEL] + [m for m in MODEL_FALLBACKS if m != SHEETFLOW_MODEL]

client: AsyncOpenAI | None = None
if OPENROUTER_API_KEY:
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        timeout=OPENROUTER_TIMEOUT,
        default_headers={
            "HTTP-Referer": os.getenv("SHEETFLOW_SITE_URL", "http://localhost:8000"),
            "X-Title": "SheetFlow AI",
        },
    )


class FormulaRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    tool_type: Literal["Excel", "Google Sheets"] = "Excel"

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Prompt cannot be empty")
        return cleaned


class _RateLimiter:
    """Simple in-memory sliding-window limiter (per process)."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = max(1, limit)
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True


rate_limiter = _RateLimiter(RATE_LIMIT_PER_MINUTE)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _friendly_error(err: Exception | None) -> str:
    if err is None:
        return "Unknown error"
    text = str(err)
    if len(text) > 220:
        return text[:217] + "..."
    return text


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "SheetFlow AI",
        "api_key_configured": bool(OPENROUTER_API_KEY),
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
    }


@app.post("/api/generate")
async def generate_formula(req: FormulaRequest, request: Request):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured. Set it in your .env file.",
        )

    ip = _client_ip(request)
    if not rate_limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Limit is {RATE_LIMIT_PER_MINUTE} per minute. Please wait and try again.",
        )

    system_prompt = f"""You are a senior spreadsheet automation specialist for {req.tool_type}.
Generate the exact, clean spreadsheet formula according to user intent.

STRICT OUTPUT FORMAT RULES:
1. First line MUST be only the raw formula starting with '=', wrapped in markdown inline code (e.g. `=SUMIF(...)`).
2. Followed by a concise bulleted explanation (max 2 points) explaining the logic and arguments.
3. No opening conversational remarks (e.g. "Sure!", "Here is your formula:"). Be purely professional and direct.
"""

    last_error: Exception | None = None
    for model in MODEL_CANDIDATES:
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req.prompt},
                ],
                temperature=0.1,
            )
            content = completion.choices[0].message.content
            if not content or not str(content).strip():
                raise RuntimeError(f"Empty response from model {model}")
            return {"result": content}
        except RateLimitError as e:
            last_error = e
            continue
        except APITimeoutError as e:
            last_error = e
            continue
        except APIError as e:
            last_error = e
            status = getattr(e, "status_code", None)
            message = str(e).lower()
            if status == 429 or "rate" in message or "temporarily" in message:
                continue
            if status in {401, 403}:
                raise HTTPException(
                    status_code=502,
                    detail="Upstream API authentication failed. Check OPENROUTER_API_KEY.",
                )
            raise HTTPException(status_code=502, detail="Upstream model provider error. Please try again.")
        except Exception as e:
            last_error = e
            message = str(e).lower()
            if "429" in message or "rate" in message or "temporarily" in message or "timeout" in message:
                continue
            raise HTTPException(status_code=500, detail="Failed to generate formula. Please try again.")

    raise HTTPException(
        status_code=429,
        detail=(
            "All formula models are temporarily unavailable or rate-limited. "
            "Please wait a moment and try again."
        ),
    )


# Mount static files last so /api/* routes stay available
if os.path.exists("index.html"):
    app.mount("/", StaticFiles(html=True, directory="."), name="static")
