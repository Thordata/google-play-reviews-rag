"""Minimal Google Play Reviews RAG using Thordata Web Scraper + OpenRouter.

Design goals:
- Single small script (no extra package layout)
- No OpenAI dependency (use OpenRouter instead)
- End‑to‑end flow: fetch reviews -> in‑memory similarity search -> chat answer

High‑level pipeline:
  1) Use Thordata Web Scraper API to fetch Google Play reviews for a single app.
  2) Load & clean the reviews into a list of short texts.
  3) Build a small embedding index in memory (NumPy array).
  4) Answer a question using OpenRouter chat + the retrieved reviews (RAG‑style).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from thordata import ThordataClient, ThordataAPIError, ThordataRateLimitError, load_env_file


# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_REVIEWS_PATH = DATA_DIR / "google_play_reviews_raw.json"

# Best effort: force UTF-8 stdout/stderr to avoid Windows GBK issues.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def load_local_env() -> None:
    """Load ./.env if present, without overriding existing environment variables."""
    # Prefer the SDK helper for consistency with other Thordata examples.
    load_env_file(str(ROOT_DIR / ".env"), override=False)
    # Also allow python-dotenv for IDE / tooling friendliness.
    load_dotenv(ROOT_DIR / ".env", override=False)


# ---------------------------------------------------------------------------
# Thordata client helpers
# ---------------------------------------------------------------------------

def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def create_thordata_client() -> ThordataClient:
    """Create a ThordataClient using environment variables.

    Required:
      - THORDATA_SCRAPER_TOKEN

    Optional (for Web Scraper Tasks):
      - THORDATA_PUBLIC_TOKEN
      - THORDATA_PUBLIC_KEY
    """
    scraper_token = _get_required_env("THORDATA_SCRAPER_TOKEN")
    public_token = os.getenv("THORDATA_PUBLIC_TOKEN")
    public_key = os.getenv("THORDATA_PUBLIC_KEY")

    return ThordataClient(
        scraper_token=scraper_token,
        public_token=public_token,
        public_key=public_key,
    )


# ---------------------------------------------------------------------------
# OpenRouter client (minimal, HTTP only)
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api"


def _get_openrouter_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Please set it in your .env.")
    return key


def openrouter_request_embeddings(texts: Sequence[str], model: str) -> np.ndarray:
    """Create embeddings via OpenRouter's OpenAI‑compatible embeddings endpoint.

    This uses the /v1/embeddings route. Many OpenRouter models provide embeddings;
    you can change the default model name in main().
    """
    api_key = _get_openrouter_api_key()

    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": list(texts),
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter embeddings request failed "
            f"({resp.status_code}): {resp.text[:400]}"
        )

    payload = resp.json()
    data = payload.get("data") or []
    if not data:
        raise RuntimeError("OpenRouter embeddings returned empty data.")

    return np.array([item["embedding"] for item in data], dtype="float32")


def openrouter_chat_completion(
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float = 0.3,
) -> str:
    """Call OpenRouter chat completions with a minimal payload."""
    api_key = _get_openrouter_api_key()

    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": list(messages),
            "temperature": float(temperature),
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter chat request failed "
            f"({resp.status_code}): {resp.text[:400]}"
        )

    payload = resp.json()
    choices = payload.get("choices") or []
    if not choices or not choices[0].get("message"):
        raise RuntimeError("OpenRouter chat returned no choices.")

    return str(choices[0]["message"]["content"])


# ---------------------------------------------------------------------------
# Google Play scraping
# ---------------------------------------------------------------------------

@dataclass
class SpiderConfig:
    spider_name: str
    spider_id: str
    app_url: str
    num_of_reviews: str
    start_date: str
    end_date: str
    country: str


def load_spider_config(app_url_override: str | None = None) -> SpiderConfig:
    """Load Google Play spider configuration from environment variables.

    An explicit app_url_override (from CLI) takes precedence over .env.
    """
    spider_name = os.getenv("GOOGLE_PLAY_SPIDER_NAME", "play.google.com")
    spider_id = os.getenv(
        "GOOGLE_PLAY_SPIDER_ID",
        "google-play-store_reviews_by-url",
    )
    if app_url_override:
        app_url = app_url_override
    else:
        app_url = _get_required_env("GOOGLE_PLAY_APP_URL")

    return SpiderConfig(
        spider_name=spider_name,
        spider_id=spider_id,
        app_url=app_url,
        num_of_reviews=os.getenv("GOOGLE_PLAY_NUM_REVIEWS", ""),
        start_date=os.getenv("GOOGLE_PLAY_START_DATE", ""),
        end_date=os.getenv("GOOGLE_PLAY_END_DATE", ""),
        country=os.getenv("GOOGLE_PLAY_COUNTRY", ""),
    )


def fetch_google_play_reviews(
    force_fetch: bool = True,
    app_url_override: str | None = None,
) -> Path:
    """Fetch Google Play reviews via Thordata Web Scraper and save to JSON."""
    if not force_fetch and RAW_REVIEWS_PATH.is_file():
        print(f"[fetch] Reusing existing raw reviews at {RAW_REVIEWS_PATH}")
        return RAW_REVIEWS_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    client = create_thordata_client()
    spider = load_spider_config(app_url_override=app_url_override)

    individual_params: Dict[str, Any] = {
        # IMPORTANT: these keys must match the spider parameter names
        "app_url": spider.app_url,
        "num_of_reviews": spider.num_of_reviews,
        "start date": spider.start_date,
        "end date": spider.end_date,
        "country": spider.country,
    }

    print("[fetch] Creating scraper task...")
    try:
        task_id = client.create_scraper_task(
            file_name="google_play_reviews",
            spider_id=spider.spider_id,
            spider_name=spider.spider_name,
            parameters=individual_params,
            universal_params=None,
        )
    except ThordataRateLimitError as exc:
        raise RuntimeError(
            "Thordata Web Scraper API reported a balance / rate-limit issue "
            "while creating the Google Play reviews task.\n"
            "Please check your Thordata account plan/balance, or run this script "
            "with --no-fetch to reuse an existing data/google_play_reviews_raw.json file."
        ) from exc
    except ThordataAPIError as exc:
        raise RuntimeError(
            f"Thordata Web Scraper API returned an error while creating the task: {exc}"
        ) from exc

    print(f"[fetch] Task created: {task_id}")

    # Poll task status
    for attempt in range(30):
        status = client.get_task_status(task_id)
        print(f"[fetch] Status attempt {attempt + 1}: {status}")
        status_l = (status or "").lower()
        if status_l in {"ready", "success", "finished"}:
            break
        if status_l in {"failed", "error"}:
            raise RuntimeError(f"Scraper task failed with status={status}")
        time.sleep(10)
    else:
        raise RuntimeError("Timeout while waiting for scraper task to finish.")

    # Get download URL & download JSON
    download_url = client.get_task_result(task_id, file_type="json")
    print(f"[fetch] Download URL: {download_url}")

    resp = requests.get(str(download_url), timeout=120)
    resp.raise_for_status()

    RAW_REVIEWS_PATH.write_text(resp.text, encoding="utf-8")
    print(f"[fetch] Saved raw reviews to {RAW_REVIEWS_PATH}")

    return RAW_REVIEWS_PATH


# ---------------------------------------------------------------------------
# Load & clean reviews
# ---------------------------------------------------------------------------

REVIEW_TEXT_FIELD = "review"          # matches Thordata Google Play example
RATING_FIELD = "review_rating"
DATE_FIELD = "review_date"


def load_reviews(path: Path) -> pd.DataFrame:
    """Load raw reviews JSON and return a cleaned DataFrame."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "data" in raw:
        records = raw["data"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Unexpected JSON structure for reviews result.")

    df = pd.DataFrame.from_records(records)

    if REVIEW_TEXT_FIELD not in df.columns:
        raise KeyError(
            f"Expected review text field '{REVIEW_TEXT_FIELD}' not found in columns: {list(df.columns)}"
        )

    cols = [c for c in [REVIEW_TEXT_FIELD, RATING_FIELD, DATE_FIELD] if c in df.columns]
    df = df[cols].rename(
        columns={
            REVIEW_TEXT_FIELD: "text",
            RATING_FIELD: "rating",
            DATE_FIELD: "date",
        }
    )

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    print(f"[load] Loaded {len(df)} reviews")
    return df


# ---------------------------------------------------------------------------
# Embeddings index (NumPy only)
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingIndex:
    embeddings: np.ndarray
    texts: List[str]
    meta: pd.DataFrame


def build_embedding_index(
    df: pd.DataFrame,
    embedding_model: str,
    max_reviews: int = 80,
    max_chars_per_review: int = 300,
) -> EmbeddingIndex:
    """Build a simple in‑memory embedding index from a DataFrame of reviews."""
    df = df.copy().reset_index(drop=True)
    if len(df) > max_reviews:
        df = df.head(max_reviews)
        print(f"[embed] Truncated to first {max_reviews} reviews for embeddings.")

    texts = [str(t)[:max_chars_per_review] for t in df["text"].tolist()]

    print(f"[embed] Creating embeddings for {len(texts)} reviews...")
    embs = openrouter_request_embeddings(texts, model=embedding_model)
    print(f"[embed] Embeddings shape: {embs.shape}")

    if "rating" in df.columns or "date" in df.columns:
        meta = df[["rating", "date"]].copy()
    else:
        meta = df[[]]

    return EmbeddingIndex(embeddings=embs, texts=texts, meta=meta)


def retrieve_top_k(
    question: str,
    index: EmbeddingIndex,
    embedding_model: str,
    top_k: int = 10,
) -> List[int]:
    """Retrieve indices of top‑k similar reviews for the question."""
    q_vec = openrouter_request_embeddings([question], model=embedding_model)[0]

    embs = index.embeddings
    norms = np.linalg.norm(embs, axis=1) * float(np.linalg.norm(q_vec))
    sims = (embs @ q_vec) / (norms + 1e-8)

    top_idx = np.argsort(-sims)[:top_k]
    return top_idx.tolist()


def build_rag_context(
    question: str,
    index: EmbeddingIndex,
    embedding_model: str,
    top_k: int,
) -> Tuple[str, List[int]]:
    """Return a context string built from top‑k reviews and the indices used."""
    top_idx = retrieve_top_k(
        question=question,
        index=index,
        embedding_model=embedding_model,
        top_k=top_k,
    )

    parts: List[str] = []
    for rank, i in enumerate(top_idx, start=1):
        text = index.texts[i]
        rating = ""
        date = ""
        if not index.meta.empty:
            rating = str(index.meta.iloc[i].get("rating", "") or "")
            date = str(index.meta.iloc[i].get("date", "") or "")

        header = f"[Review {rank}]"
        if rating or date:
            header += f" rating={rating or 'NA'}, date={date or 'NA'}"

        parts.append(f"{header}\n{text}")

    context = "\n\n".join(parts)
    return context, top_idx


def answer_question_with_rag(
    question: str,
    index: EmbeddingIndex,
    embedding_model: str,
    chat_model: str,
    top_k: int = 10,
) -> str:
    """Use RAG over reviews to answer a question with OpenRouter."""
    context, _ = build_rag_context(
        question=question,
        index=index,
        embedding_model=embedding_model,
        top_k=top_k,
    )

    system_prompt = (
        "You are a mobile app product analyst.\n"
        "You are given a set of Google Play reviews for one app.\n"
        "Answer the user's question ONLY based on these reviews.\n"
        "Highlight recurring themes and include a few short, concrete examples.\n"
        "If the reviews do not contain enough information, say so explicitly."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Here are representative user reviews:\n{context}\n\n"
        "Please answer in a concise way (bullet points are welcome) and focus on:\n"
        "- overall sentiment\n"
        "- most common complaints\n"
        "- most common praises\n"
    )

    return openrouter_chat_completion(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Google Play Reviews RAG with Thordata + OpenRouter",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch latest reviews from Thordata Web Scraper API.",
    )
    parser.add_argument(
        "--no-fetch",
        dest="fetch",
        action="store_false",
        help="Skip fetching and reuse cached data/google_play_reviews_raw.json.",
    )
    parser.set_defaults(fetch=True)

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask about the app reviews.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of reviews to retrieve for RAG (default: 10).",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG: only fetch/load and show basic stats (no LLM calls).",
    )
    parser.add_argument(
        "--app-url",
        type=str,
        help=(
            "Override GOOGLE_PLAY_APP_URL from .env for this run. "
            "If not provided, the value from .env is used."
        ),
    )
    parser.add_argument(
        "--app-urls",
        type=str,
        help=(
            "Comma-separated list of app URLs to process sequentially. "
            "If provided with --fetch, the script will fetch and analyze each app in turn."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help=(
            "Embedding model name for OpenRouter's /v1/embeddings endpoint "
            "(default: text-embedding-3-small, must be available via OpenRouter)."
        ),
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default="stepfun/step-3.5-flash:free",
        help=(
            "Chat model name for OpenRouter (default: stepfun/step-3.5-flash:free). "
            "You can also try: openrouter/free, qwen/qwen3-coder:free, etc."
        ),
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=80,
        help="Maximum number of reviews to embed (default: 80).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=300,
        help="Maximum characters per review text used for embeddings (default: 300).",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        help=(
            "Optional path to save a Markdown report with the question and RAG answer. "
            "The answer is still printed to the console."
        ),
    )

    return parser


def main() -> None:
    load_local_env()

    parser = build_arg_parser()
    args = parser.parse_args()

    # When using multiple app URLs, require fetch for more than one app
    app_urls: list[str] = []
    if args.app_urls:
        app_urls.extend(u.strip() for u in str(args.app_urls).split(",") if u.strip())
    # Single override can be treated uniformly as a list of length 1
    if args.app_url and not app_urls:
        app_urls.append(args.app_url)

    if app_urls and len(app_urls) > 1 and not args.fetch:
        raise RuntimeError(
            "When using --app-urls with multiple apps, --fetch is required so that "
            "each app has its own fresh reviews."
        )

    # Helper to run the core flow for a single app URL override (or default env value).
    def run_for_single_app(app_url_override: str | None) -> None:
        # 1) Fetch reviews if requested
        if args.fetch:
            path = fetch_google_play_reviews(
                force_fetch=True,
                app_url_override=app_url_override,
            )
        else:
            if not RAW_REVIEWS_PATH.is_file():
                raise FileNotFoundError(
                    f"{RAW_REVIEWS_PATH} does not exist. "
                    "Run once with --fetch to create it, or drop your own JSON there."
                )
            path = RAW_REVIEWS_PATH

        # 2) Load & clean
        df = load_reviews(path)

        if args.no_rag:
            print(
                "\n[summary] RAG disabled by --no-rag. "
                "Showing basic review stats.\n"
            )
            print(df.head())
            print("\n[rating distribution]")
            if "rating" in df.columns:
                print(df["rating"].value_counts().sort_index())
            return

        # 3) Build embeddings index
        index = build_embedding_index(
            df=df,
            embedding_model=args.embedding_model,
            max_reviews=max(1, args.max_reviews),
            max_chars_per_review=max(1, args.max_chars),
        )

        # 4) RAG QA via OpenRouter
        print(f"\n[qa] Answering question: {args.question!r}\n")
        answer = answer_question_with_rag(
            question=args.question,
            index=index,
            embedding_model=args.embedding_model,
            chat_model=args.chat_model,
            top_k=args.top_k,
        )

        print("\n=== RAG Answer ===\n")
        print(answer)

        # Optional Markdown report for easier sharing.
        if args.output_markdown:
            # If multiple apps are processed, treat output_markdown as a directory.
            output_path = Path(args.output_markdown)
            if app_urls and len(app_urls) > 1:
                output_dir = output_path
                output_dir.mkdir(parents=True, exist_ok=True)
                slug = (
                    app_url_override
                    if app_url_override
                    else os.getenv("GOOGLE_PLAY_APP_URL", "default")
                )
                slug = slug.replace("https://", "").replace("http://", "")
                slug = slug.replace("/", "_").replace("?", "_").replace("&", "_")
                report_path = output_dir / f"{slug}.md"
            else:
                report_path = output_path
                report_path.parent.mkdir(parents=True, exist_ok=True)

            meta_lines = [
                f"- App URL: {app_url_override or os.getenv('GOOGLE_PLAY_APP_URL')}",
                f"- Reviews embedded: {len(index.texts)} (top_k={args.top_k})",
                f"- Embedding model: {args.embedding_model}",
                f"- Chat model: {args.chat_model}",
            ]
            md = (
                f"# Google Play Reviews RAG Report\n\n"
                f"**Question**\n\n"
                f"{args.question}\n\n"
                f"**Metadata**\n\n"
                + "\n".join(meta_lines)
                + "\n\n"
                f"**Answer**\n\n"
                f"{answer}\n"
            )
            report_path.write_text(md, encoding="utf-8")
            print(f"\n[report] Saved Markdown report to {report_path}")

    # If multiple app URLs are provided, process each sequentially.
    if app_urls:
        for url in app_urls:
            print("\n" + "=" * 80)
            print(f"[app] Processing app: {url}")
            print("=" * 80)
            run_for_single_app(app_url_override=url)
        return

    # Fallback: single-app flow using .env defaults (and optional --app-url override).
    run_for_single_app(app_url_override=args.app_url)


if __name__ == "__main__":
    main()