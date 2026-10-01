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

# 读取环境变量中的 API Key（兼容你 Render 现有的变量名）
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
SHEETFLOW_MODEL = os.getenv("SHEETFLOW_MODEL", "gemini-3.8-flash").strip()
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "15"))
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "2000"))
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "45"))

client: AsyncOpenAI | None = None
if OPENROUTER_API_KEY:
    client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=OPENROUTER_API_KEY,
        timeout=OPENROUTER_TIMEOUT,
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
            detail="API Key is not configured in Render environment variables.",
        )

    ip = _client_ip(request)
    if not rate_limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Limit is {RATE_LIMIT_PER_MINUTE} per minute. Please wait a moment.",
        )

    system_prompt = f"""You are a senior spreadsheet automation specialist for {req.tool_type}.
Generate the exact, clean spreadsheet formula according to user intent.

STRICT OUTPUT FORMAT RULES:
1. First line MUST be only the raw formula starting with '=', wrapped in markdown inline code (e.g. `=SUMIF(...)`).
2. Followed by a concise bulleted explanation (max 2 points) explaining the logic and arguments.
3. No opening conversational remarks (e.g. "Sure!", "Here is your formula:"). Be purely professional and direct.
"""

    try:
        completion = await client.chat.completions.create(
            model=SHEETFLOW_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.prompt},
            ],
            temperature=0.1,
        )
        content = completion.choices[0].message.content
        if not content or not str(content).strip():
            raise RuntimeError("Empty response received from Gemini.")
        return {"result": content}

    except RateLimitError as e:
        raise HTTPException(
            status_code=429,
            detail="Gemini API rate limit reached. Please wait a few seconds and try again.",
        )
    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request to Gemini timed out. Please try again.",
        )
    except APIError as e:
        # 直接输出真实的 Google 错误详情，方便立刻定位
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API Error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Server Error: {str(e)}",
        )


# Mount static files last so /api/* routes stay available
if os.path.exists("index.html"):
    app.mount("/", StaticFiles(html=True, directory="."), name="static")