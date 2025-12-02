"""
Google Play Reviews RAG example with Thordata Web Scraper API + OpenAI.

Pipeline:
  1) Use Thordata Web Scraper API to fetch Google Play reviews for one app.
  2) Load and clean the reviews (text, rating, date).
  3) Build a small embedding index with OpenAI.
  4) Answer questions about the app using retrieved reviews (RAG-style).

IMPORTANT:
  - Spider ID / name / parameters are placeholders; you must update them
    according to your Thordata Dashboard configuration.

Usage (fetch latest reviews + build RAG index + answer a question):

    python google_play_reviews_rag.py --fetch \
      --question "What do users complain about most?"

Usage (re-use cached JSON in data/google_play_reviews_raw.json, only RAG):

    python google_play_reviews_rag.py \
      --question "What do users like about the app UI?" --no-fetch
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError, RateLimitError

from thordata import ThordataClient

# -----------------------------
# Paths & environment
# -----------------------------

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
DATA_DIR = ROOT_DIR / "data"
RAW_REVIEWS_PATH = DATA_DIR / "google_play_reviews_raw.json"

load_dotenv(ENV_PATH, override=True)

# -----------------------------
# Thordata client
# -----------------------------

SCRAPER_TOKEN = os.getenv("THORDATA_SCRAPER_TOKEN")
PUBLIC_TOKEN = os.getenv("THORDATA_PUBLIC_TOKEN")
PUBLIC_KEY = os.getenv("THORDATA_PUBLIC_KEY")

if not SCRAPER_TOKEN:
    raise RuntimeError(
        "THORDATA_SCRAPER_TOKEN is missing. "
        "Please configure your .env file at the project root."
    )

td_client = ThordataClient(
    scraper_token=SCRAPER_TOKEN,
    public_token=PUBLIC_TOKEN,
    public_key=PUBLIC_KEY,
)

# -----------------------------
# Google Play Spider settings
# -----------------------------

# From Thordata Dashboard / official example:
#   spider_name = "play.google.com"
#   spider_id   = "google-play-store_reviews_by-url"
GOOGLE_PLAY_SPIDER_ID = os.getenv("GOOGLE_PLAY_SPIDER_ID", "google-play-store_reviews_by-url")
GOOGLE_PLAY_SPIDER_NAME = os.getenv("GOOGLE_PLAY_SPIDER_NAME", "play.google.com")

GOOGLE_PLAY_APP_URL = os.getenv("GOOGLE_PLAY_APP_URL")  # e.g. "https://play.google.com/store/apps/details?id=com.linkedin.android")
GOOGLE_PLAY_NUM_REVIEWS = os.getenv("GOOGLE_PLAY_NUM_REVIEWS", "")
GOOGLE_PLAY_START_DATE = os.getenv("GOOGLE_PLAY_START_DATE", "")
GOOGLE_PLAY_END_DATE = os.getenv("GOOGLE_PLAY_END_DATE", "")
GOOGLE_PLAY_COUNTRY = os.getenv("GOOGLE_PLAY_COUNTRY", "")

if not GOOGLE_PLAY_SPIDER_ID or not GOOGLE_PLAY_APP_URL:
    raise RuntimeError(
        "GOOGLE_PLAY_SPIDER_ID and GOOGLE_PLAY_APP_URL must be set in your .env.\n"
        "Please check your Thordata Dashboard for the correct spider id/name and parameters."
    )

# These keys MUST match the Thordata Spider parameters exactly
INDIVIDUAL_PARAMS: Dict[str, Any] = {
    "app_url": GOOGLE_PLAY_APP_URL,
    "num_of_reviews": GOOGLE_PLAY_NUM_REVIEWS,
    "start date": GOOGLE_PLAY_START_DATE,
    "end date": GOOGLE_PLAY_END_DATE,
    "country": GOOGLE_PLAY_COUNTRY,
}


# -----------------------------
# OpenAI client
# -----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Please set it in your .env file."
    )

openai_client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


# -----------------------------
# Step 1: Fetch reviews via Thordata Web Scraper API
# -----------------------------

def fetch_google_play_reviews(force_fetch: bool = True) -> Path:
    """
    Fetch Google Play reviews via Thordata Web Scraper API and save to JSON.

    Returns:
        Path to RAW_REVIEWS_PATH.
    """
    if not force_fetch and RAW_REVIEWS_PATH.is_file():
        print(f"[fetch] Reusing existing raw reviews at {RAW_REVIEWS_PATH}")
        return RAW_REVIEWS_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[fetch] Creating scraper task...")
    try:
        task_id = td_client.create_scraper_task(
            file_name="google_play_reviews",
            spider_id=GOOGLE_PLAY_SPIDER_ID,
            spider_name=GOOGLE_PLAY_SPIDER_NAME,
            individual_params=INDIVIDUAL_PARAMS,
            universal_params=None,
        )
    except Exception as e:
        msg = str(e)
        if "code': 402" in msg or "Insufficient permissions" in msg:
            raise RuntimeError(
                "Thordata Web Scraper API returned code 402 "
                "(insufficient permissions / balance). "
                "Please check your Thordata account balance or plan, "
                "or run this script with --no-fetch to reuse an existing "
                "data/google_play_reviews_raw.json file."
            ) from e
        raise
    task_id = td_client.create_scraper_task(
        file_name="google_play_reviews",
        spider_id=GOOGLE_PLAY_SPIDER_ID,
        spider_name=GOOGLE_PLAY_SPIDER_NAME,
        individual_params=INDIVIDUAL_PARAMS,
        universal_params=None,
    )
    print(f"[fetch] Task created: {task_id}")

    # Poll status
    for attempt in range(30):
        status = td_client.get_task_status(task_id)
        print(f"[fetch] Status attempt {attempt+1}: {status}")
        if status.lower() in {"ready", "success", "finished"}:
            break
        if status.lower() in {"failed", "error"}:
            raise RuntimeError(f"Scraper task failed with status={status}")
        time.sleep(10)
    else:
        raise RuntimeError("Timeout while waiting for scraper task to finish.")

    # Get download URL
    download_url = td_client.get_task_result(task_id, file_type="json")
    print(f"[fetch] Download URL: {download_url}")

    # Download results via direct HTTP GET
    resp = requests.get(download_url, timeout=60)
    resp.raise_for_status()

    # NOTE: we assume the result is JSON. If your spider is configured to return
    # CSV/ZIP, adjust this part accordingly.
    with RAW_REVIEWS_PATH.open("w", encoding="utf-8") as f:
        f.write(resp.text)

    print(f"[fetch] Saved raw reviews to {RAW_REVIEWS_PATH}")
    return RAW_REVIEWS_PATH


# -----------------------------
# Step 2: Load & clean reviews
# -----------------------------

# If your spider uses different field names, update these constants.
REVIEW_TEXT_FIELD = "review_text"    # <-- confirm with your spider schema
RATING_FIELD = "rating"
DATE_FIELD = "review_date"


def load_reviews_from_json(path: Path) -> pd.DataFrame:
    """
    Load raw reviews JSON and return a cleaned DataFrame.

    Expected JSON structure:
      - Either a list of objects: [{...}, {...}, ...]
      - Or an object with a "data" field: {"data": [{...}, ...]}

    Adjust this function according to your actual spider output.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        records = data["data"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unexpected JSON structure for reviews result.")

    df = pd.DataFrame.from_records(records)

    if REVIEW_TEXT_FIELD not in df.columns:
        raise KeyError(
            f"Expected review text field '{REVIEW_TEXT_FIELD}' not found in columns: {df.columns}"
        )

    # Keep only a few useful columns
    cols = [c for c in [REVIEW_TEXT_FIELD, RATING_FIELD, DATE_FIELD] if c in df.columns]
    df = df[cols].rename(
        columns={
            REVIEW_TEXT_FIELD: "text",
            RATING_FIELD: "rating",
            DATE_FIELD: "date",
        }
    )

    # Basic cleaning
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    print(f"[load] Loaded {len(df)} reviews")
    return df


# -----------------------------
# Step 3: Build embeddings index
# -----------------------------

def build_embeddings(df: pd.DataFrame, max_reviews: int = 80) -> Dict[str, Any]:
    """
    Build embeddings for a subset of reviews. Returns a dict with:

        {
          "embeddings": np.ndarray of shape (N, D),
          "texts": List[str],
          "meta":  pd.DataFrame with rating/date for each text,
        }

    We keep a small number of reviews to stay within TPM / RPM limits.
    """
    df = df.copy().reset_index(drop=True)
    if len(df) > max_reviews:
        df = df.head(max_reviews)
        print(f"[embed] Truncated to first {max_reviews} reviews for embeddings.")

    texts = df["text"].tolist()

    # Truncate each text to avoid huge tokens
    MAX_CHARS_PER_REVIEW = 300
    texts = [t[:MAX_CHARS_PER_REVIEW] for t in texts]

    print(f"[embed] Creating embeddings for {len(texts)} reviews...")
    try:
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
    except RateLimitError as e:
        raise RuntimeError(
            "OpenAI returned 'insufficient_quota' for embeddings.\n"
            "Please check your OpenAI plan/billing, or disable the RAG part\n"
            "of this script before running again."
        ) from e
    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    embs = np.array([item.embedding for item in resp.data], dtype="float32")
    print(f"[embed] Embeddings shape: {embs.shape}")

    meta = df[["rating", "date"]] if "rating" in df.columns else df[[]]

    return {
        "embeddings": embs,
        "texts": texts,
        "meta": meta,
    }


# -----------------------------
# Step 4: RAG-style QA
# -----------------------------

def retrieve_top_k(
    question: str,
    index: Dict[str, Any],
    top_k: int = 10,
) -> List[int]:
    """
    Retrieve top-k most similar reviews for the given question.
    """
    q_resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    )
    q_vec = np.array(q_resp.data[0].embedding, dtype="float32")

    embs = index["embeddings"]
    # Cosine similarity
    norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(q_vec)
    sims = (embs @ q_vec) / (norms + 1e-8)

    top_idx = np.argsort(-sims)[:top_k]
    return top_idx.tolist()


def answer_question_with_rag(
    question: str,
    index: Dict[str, Any],
    top_k: int = 10,
) -> str:
    """
    Build a context from top-k reviews and ask OpenAI for an answer.
    """
    top_idx = retrieve_top_k(question, index, top_k=top_k)

    texts = index["texts"]
    meta = index["meta"]

    context_parts = []
    for rank, i in enumerate(top_idx, start=1):
        text = texts[i]
        if not meta.empty:
            rating = meta.iloc[i].get("rating", "")
            date = meta.iloc[i].get("date", "")
        else:
            rating, date = "", ""

        context_parts.append(
            f"[Review {rank}] rating={rating}, date={date}\n{text}\n"
        )

    context_text = "\n\n".join(context_parts)

    system_prompt = (
        "You are a product analyst for a mobile app.\n"
        "You are given a set of Google Play reviews for this app.\n"
        "Answer the user's question ONLY based on these reviews.\n"
        "Highlight recurring themes and include concrete examples."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Here are some representative reviews:\n{context_text}\n\n"
        "Please provide a concise answer (bullet points are OK), "
        "focusing on user sentiment, common complaints, and praises."
    )

    try:
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
    except RateLimitError as e:
        raise RuntimeError(
            "OpenAI returned 'insufficient_quota' for chat completions.\n"
            "Please check your OpenAI plan/billing."
        ) from e

    return resp.choices[0].message.content


# -----------------------------
# CLI entrypoint
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google Play Reviews RAG with Thordata + OpenAI"
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
        help="Skip embeddings + QA, only fetch/load and show basic stats.",
    )

    args = parser.parse_args()

    # 1) Fetch reviews if requested
    if args.fetch:
        path = fetch_google_play_reviews(force_fetch=True)
    else:
        if not RAW_REVIEWS_PATH.is_file():
            raise FileNotFoundError(
                f"{RAW_REVIEWS_PATH} does not exist. "
                "Run with --fetch at least once to create it."
            )
        path = RAW_REVIEWS_PATH

    # 2) Load & clean
    df = load_reviews_from_json(path)

    # 如果用户只想抓数据，不想跑 RAG（或者没 OpenAI 配额）
    if args.no_rag:
        print("\n[summary] Showing basic review stats (RAG disabled by --no-rag)\n")
        print(df.head())
        print("\nRating distribution:")
        if "rating" in df.columns:
            print(df["rating"].value_counts().sort_index())
        return

    # 3) Build embeddings index
    index = build_embeddings(df, max_reviews=80)

    # 4) RAG QA
    print(f"\n[qa] Answering question: {args.question!r}\n")
    answer = answer_question_with_rag(args.question, index, top_k=args.top_k)

    print("\n=== RAG Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()