## Google Play Reviews RAG (Thordata + OpenRouter)

Minimal, end‑to‑end example to analyze **Google Play app reviews** using:

- **Thordata Web Scraper API** to collect reviews
- **OpenRouter** (free‑tier friendly) for embeddings + chat
- A tiny, in‑memory RAG pipeline in a single script

Everything in this repo is kept as small as possible: one main script, one `.env`, one `requirements.txt`.

---

### Requirements

- Python **3.10+**
- A Thordata account and credentials:
  - `THORDATA_SCRAPER_TOKEN` (required)
  - `THORDATA_PUBLIC_TOKEN` / `THORDATA_PUBLIC_KEY` (required for Web Scraper tasks)
- Access to the Google Play reviews spider, for example:
  - `spider_name`: `play.google.com`
  - `spider_id`: `google-play-store_reviews_by-url`
- An OpenRouter API key (`OPENROUTER_API_KEY`)

---

### Install

```bash
cd google-play-reviews-rag

python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS / Linux

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

### Configure `.env`

Create your own `.env` from the example:

```bash
cp .env.example .env
```

Fill in at least these values:

```env
# Thordata
THORDATA_SCRAPER_TOKEN=...
THORDATA_PUBLIC_TOKEN=...
THORDATA_PUBLIC_KEY=...

# OpenRouter
OPENROUTER_API_KEY=sk-or-...

# Google Play spider
GOOGLE_PLAY_SPIDER_ID=google-play-store_reviews_by-url
GOOGLE_PLAY_SPIDER_NAME=play.google.com

GOOGLE_PLAY_APP_URL=https://play.google.com/store/apps/details?id=com.linkedin.android
GOOGLE_PLAY_NUM_REVIEWS=
GOOGLE_PLAY_START_DATE=
GOOGLE_PLAY_END_DATE=
GOOGLE_PLAY_COUNTRY=
```

The individual parameter keys used in `google_play_reviews_rag.py` are:

```python
{
    "app_url": GOOGLE_PLAY_APP_URL,
    "num_of_reviews": GOOGLE_PLAY_NUM_REVIEWS,
    "start date": GOOGLE_PLAY_START_DATE,
    "end date": GOOGLE_PLAY_END_DATE,
    "country": GOOGLE_PLAY_COUNTRY,
}
```

They **must match** the parameter names of your Thordata spider.

---

### Quick start: fetch + RAG in one command

Fetch the latest reviews and run a simple RAG analysis in one shot:

```bash
python google_play_reviews_rag.py --fetch ^
  --question "What do users complain about most?" ^
  --top-k 10
```

What this does:

1. Creates a Thordata Web Scraper task for the app.
2. Waits for completion and downloads JSON results to `data/google_play_reviews_raw.json`.
3. Cleans the reviews into a small DataFrame.
4. Builds an in‑memory embedding index (NumPy only).
5. Uses OpenRouter to answer the question based on the top‑K similar reviews.

---

### Reuse existing JSON (no new scraping)

Once `data/google_play_reviews_raw.json` exists, you can skip the scraping step:

```bash
python google_play_reviews_rag.py --no-fetch ^
  --question "What do users like about the app UI?" ^
  --top-k 10
```

This is useful when:

- You want to iterate on the RAG / prompt logic without re-running the spider.
- You want to avoid extra Web Scraper usage during local development.

---

### Data‑only mode (no LLM, no embeddings)

If you just want to fetch and inspect raw reviews, disable RAG:

```bash
# Fetch from Thordata, then show basic stats only
python google_play_reviews_rag.py --fetch \
  --question "Anything" \
  --no-rag

# Reuse cached JSON, show stats only
python google_play_reviews_rag.py --no-fetch \
  --question "Anything" \
  --no-rag
```

In this mode the script:

- Loads reviews from JSON into a DataFrame.
- Prints the first few rows.
- Prints a rating distribution (if available).
- Does **not** call OpenRouter.

---

### Models

By default the script uses:

- Embeddings: `text-embedding-3-small` (via OpenRouter `/v1/embeddings`)
- Chat: `stepfun/step-3.5-flash:free` (via OpenRouter `/v1/chat/completions`)

You can override them from the CLI:

```bash
python google_play_reviews_rag.py --fetch ^
  --question "Summarize overall user sentiment" ^
  --embedding-model text-embedding-3-small ^
  --chat-model openrouter/free
```

Make sure the chosen models are available in your OpenRouter account.

---

### Advanced options (still minimal)

All options are **optional** – you can ignore them for a quick start.

- **Override app URL at runtime**  

  ```bash
  python google_play_reviews_rag.py --fetch ^
    --app-url "https://play.google.com/store/apps/details?id=com.whatsapp" ^
    --question "What do users like most?" ^
    --top-k 10
  ```

  This does not modify your `.env`; it only affects the current run.

- **Control how much data goes into embeddings**

  ```bash
  python google_play_reviews_rag.py --no-fetch ^
    --question "Top UX issues" ^
    --max-reviews 50 ^
    --max-chars 400
  ```

  - `--max-reviews`: maximum number of reviews to embed (default: 80).  
  - `--max-chars`: maximum characters per review text (default: 300).

- **Save a Markdown report**

  ```bash
  python google_play_reviews_rag.py --no-fetch ^
    --question "Summarize pricing and subscription related feedback" ^
    --top-k 10 ^
    --output-markdown reports/pricing_feedback.md
  ```

  The answer is still printed to the console, and a simple shareable report is saved under `reports/`.

- **Analyze multiple apps in one go**

  ```bash
  python google_play_reviews_rag.py --fetch ^
    --app-urls "https://play.google.com/store/apps/details?id=com.linkedin.android,https://play.google.com/store/apps/details?id=com.whatsapp" ^
    --question "High-level user sentiment" ^
    --top-k 5 ^
    --output-markdown reports/multi_app
  ```

  - Each app is fetched and analyzed sequentially.
  - Console will show separate sections per app.
  - Markdown reports will be written under `reports/multi_app/`, one file per app.
```

Make sure the chosen models are available in your OpenRouter account.

---

### Project layout

```text
google-play-reviews-rag/
  google_play_reviews_rag.py   # main script (scrape + clean + RAG)
  requirements.txt             # minimal dependencies
  .env.example                 # example configuration
  .env                         # your local credentials (not committed)
  data/
    google_play_reviews_raw.json  # raw JSON from Web Scraper (runtime only)
```

The `data/` directory is used only for local development and caching and is ignored by Git.

---

### Common issues

- **Thordata returns code 402 / rate limit**  
  Check your Thordata plan and balance. For local RAG testing, you can reuse a previously downloaded JSON with `--no-fetch`.

- **OpenRouter returns an error (e.g., 401 / 429)**  
  Check `OPENROUTER_API_KEY` and your OpenRouter dashboard. You can always run in `--no-rag` mode to debug the scraping part first.

---

### License

This example is provided for educational purposes and is licensed under the MIT License. See `LICENSE` for details.