# Google Play Reviews RAG with Thordata

This repository shows how to build a small **RAG‑style analysis tool** for a single Google Play app:

1. Use **Thordata Web Scraper API** to fetch user reviews from Google Play.
2. Clean and structure the reviews into a Pandas DataFrame.
3. Use **OpenAI embeddings** to build a simple similarity index.
4. Ask questions about the app based on real user reviews (RAG).

It also supports a **data‑only mode**, where you only fetch and inspect reviews (without calling OpenAI).

> Note: This example assumes you have access to the Thordata Google Play reviews spider, typically:
>
> - `spider_name`: `play.google.com`  
> - `spider_id`:   `google-play-store_reviews_by-url`

---

## 🧩 Requirements

- Python **3.10+** (3.11 recommended)
- A Thordata account and API credentials:
  - `THORDATA_SCRAPER_TOKEN`
  - `THORDATA_PUBLIC_TOKEN`
  - `THORDATA_PUBLIC_KEY`
- Access to a Thordata Web Scraper spider that fetches Google Play reviews (e.g. `google-play-store_reviews_by-url`).
- An OpenAI API key (`OPENAI_API_KEY`) for embeddings and chat.

---

## 📦 Installation

Clone this repository and create a virtual environment:

```bash
git clone https://github.com/Thordata/google-play-reviews-rag.git
cd google-play-reviews-rag

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🔐 Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Thordata credentials
THORDATA_SCRAPER_TOKEN=your_thordata_scraper_token
THORDATA_PUBLIC_TOKEN=your_thordata_public_token
THORDATA_PUBLIC_KEY=your_thordata_public_key

# OpenAI
OPENAI_API_KEY=sk-...

# Google Play Spider (from Thordata Dashboard)
GOOGLE_PLAY_SPIDER_ID=google-play-store_reviews_by-url
GOOGLE_PLAY_SPIDER_NAME=play.google.com

# Individual spider parameters (match Spider config exactly)
GOOGLE_PLAY_APP_URL=https://play.google.com/store/apps/details?id=com.linkedin.android
GOOGLE_PLAY_NUM_REVIEWS=
GOOGLE_PLAY_START_DATE=
GOOGLE_PLAY_END_DATE=
GOOGLE_PLAY_COUNTRY=
```

The keys in `INDIVIDUAL_PARAMS` inside `google_play_reviews_rag.py` must match your spider's parameter names exactly:

```python
INDIVIDUAL_PARAMS = {
    "app_url": GOOGLE_PLAY_APP_URL,
    "num_of_reviews": GOOGLE_PLAY_NUM_REVIEWS,
    "start date": GOOGLE_PLAY_START_DATE,
    "end date": GOOGLE_PLAY_END_DATE,
    "country": GOOGLE_PLAY_COUNTRY,
}
```

---

## 🚀 Usage

The main script is `google_play_reviews_rag.py`. It has three main phases:

1. Fetch reviews via Thordata Web Scraper API (optional).
2. Load & clean the reviews from JSON into a DataFrame.
3. RAG QA using OpenAI embeddings + chat (optional if you use `--no-rag`).

### 1. Fetch latest reviews + run RAG

```bash
python google_play_reviews_rag.py --fetch \
  --question "What do users complain about most?" \
  --top-k 10
```

This will:

1. Create a Web Scraper task for Google Play reviews.
2. Wait until it finishes and download the JSON result.
3. Save raw data to `data/google_play_reviews_raw.json`.
4. Load & clean the reviews.
5. Build embeddings for a subset of reviews.
6. Retrieve the top‑K relevant reviews and ask OpenAI for an answer.

### 2. Reuse cached JSON (no fetch) + run RAG

Once `data/google_play_reviews_raw.json` exists, you can skip the fetch step:

```bash
python google_play_reviews_rag.py --no-fetch \
  --question "What do users like about the app UI?" \
  --top-k 10
```

This is useful when:

- You want to iterate on the RAG logic without hitting Thordata again.
- You want to avoid Web Scraper API usage during local development.

### 3. Data‑only mode (no embeddings / no OpenAI)

If you only want to fetch and inspect reviews (without RAG or OpenAI), use the `--no-rag` flag:

```bash
# Fetch from Thordata, then show basic stats only
python google_play_reviews_rag.py --fetch \
  --question "Anything" \
  --no-rag

# Reuse existing JSON, show stats only
python google_play_reviews_rag.py --no-fetch \
  --question "Anything" \
  --no-rag
```

This mode will:

- Load the reviews into a DataFrame.
- Print a sample of the reviews.
- Print a basic rating distribution (if available).
- No OpenAI calls are made in this mode.

---

## ⚙️ How it works

### 1. Fetching reviews

`fetch_google_play_reviews()` calls:

- `ThordataClient.create_scraper_task(...)`
- `ThordataClient.get_task_status(...)`
- `ThordataClient.get_task_result(...)`

and then downloads the resulting JSON file to:

```
data/google_play_reviews_raw.json
```

If the Thordata backend returns code 402 (insufficient permissions / balance), the script will raise a clear error instructing the user to check their plan/billing.

### 2. Loading & cleaning

`load_reviews_from_json(path)` expects either:

- A list of review objects: `[{...}, {...}, ...]`, or
- An object with a "data" field: `{"data": [{...}, ...]}`.

It then:

- Extracts `review_text`, `rating`, `review_date` (configurable via constants at the top of the script).
- Renames them to `text`, `rating`, `date`.
- Drops empty texts and resets the index.

### 3. Embeddings index

`build_embeddings(df, max_reviews=80)`:

- Truncates to at most `max_reviews` reviews.
- Truncates each review text to a safe length to control tokens.
- Calls OpenAI embeddings API (`text-embedding-3-small`) to create vectors.

Returns:

```python
{
    "embeddings": np.ndarray,  # shape (N, D)
    "texts": List[str],
    "meta":  pd.DataFrame,     # rating/date per review (if available)
}
```

If OpenAI returns `insufficient_quota`, the script raises a human‑readable RuntimeError explaining the situation.

### 4. RAG QA

`answer_question_with_rag(question, index, top_k)`:

- Embeds the question.
- Retrieves the top‑K most similar reviews by cosine similarity.
- Builds a context string summarizing those reviews (rating/date + text).
- Calls OpenAI Chat Completions (`gpt-4o-mini`) to answer the question:
  - Focus on user complaints / praises.
  - Highlight recurring themes and concrete examples.

---

## 📂 Project structure

```
google-play-reviews-rag/
├── google_play_reviews_rag.py   # main script (fetch + clean + RAG)
├── requirements.txt             # Python dependencies
├── .env.example                 # example configuration (copy to .env)
├── .gitignore
└── data/
    └── google_play_reviews_raw.json   # raw JSON from Web Scraper API (runtime only)
```

The `data/` directory is ignored by Git and is used only for local development and caching.

---

## 📝 Common issues

### Thordata returns code 402 (insufficient permissions / balance)

Check your Thordata account balance / plan, or contact support.  
You can still use `--no-fetch` with a local JSON file for development.

### OpenAI returns insufficient_quota (429)

The script will raise a clear RuntimeError. In that case you can:

- Use `--no-rag` to only fetch + inspect reviews, or
- Upgrade your OpenAI plan and re‑run the RAG part.

---

## 🧾 License

This example is provided for educational purposes and is licensed under the MIT License. See the LICENSE file for details.