import os
import re
import time
import pickle
import numpy as np
import faiss
import requests
from urllib.parse import unquote
from dotenv import load_dotenv
from google.cloud import bigquery
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PROJECT_ID    = os.getenv("BIGQUERY_PROJECT_ID")
DATASET       = os.getenv("BIGQUERY_DATASET")
INDEX_PATH    = "data/text_index.faiss"
METADATA_PATH = "data/text_metadata.pkl"

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


def _fetch_wikipedia(page_title: str) -> str:
    """Fetch plain text content of a Wikipedia page via API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action":      "query",
        "titles":      page_title,
        "prop":        "extracts",
        "explaintext": True,
        "format":      "json",
    }
    try:
        headers = {"User-Agent": "PitWall-F1/1.0 (ankithreddy614@gmail.com)"}
        resp    = requests.get(url, params=params, headers=headers, timeout=10)
        data    = resp.json()
        pages   = data["query"]["pages"]
        page    = next(iter(pages.values()))

        # -1 means Wikipedia returned a missing page
        if list(pages.keys())[0] == "-1":
            return ""

        text = page.get("extract", "")
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    except Exception as e:
        print(f"    WARNING: could not fetch {page_title}: {e}")
        return ""


def _extract_title(url: str) -> str | None:
    """Extract and decode Wikipedia page title from a URL."""
    match = re.search(r'/wiki/(.+)$', url)
    return unquote(match.group(1)) if match else None


def _get_pages_from_bigquery(top_n_drivers: int = 20) -> list[str]:
    """
    Pull Wikipedia page titles from BigQuery.
    Returns:
      - Top N drivers by all-time wins (historical depth)
      - All drivers who raced in 2022+ (current grid coverage)
      - Top 5 constructors by wins
    Deduplicates automatically.
    """
    client     = bigquery.Client(project=PROJECT_ID)
    page_titles = []
    seen        = set()

    def _add(title: str, label: str):
        if title and title not in seen:
            seen.add(title)
            page_titles.append(title)
            print(f"    + {label} → {title}")

    # 1. Top drivers by all-time wins
    print("  Top winners (all time)...")
    driver_sql = f"""
        SELECT d.forename, d.surname, d.url, COUNT(*) AS wins
        FROM `{PROJECT_ID}.{DATASET}.results` r
        JOIN `{PROJECT_ID}.{DATASET}.drivers` d ON r.driverId = d.driverId
        WHERE r.position = 1
          AND d.url IS NOT NULL AND d.url != ''
        GROUP BY d.forename, d.surname, d.url
        ORDER BY wins DESC
        LIMIT {top_n_drivers}
    """
    for row in client.query(driver_sql).result():
        _add(_extract_title(row.url), f"{row.forename} {row.surname} ({row.wins} wins)")

    # 2. Current grid — anyone who raced in 2022 or later
    print("  Current grid (2022+)...")
    recent_sql = f"""
        SELECT DISTINCT d.forename, d.surname, d.url
        FROM `{PROJECT_ID}.{DATASET}.results` r
        JOIN `{PROJECT_ID}.{DATASET}.drivers` d ON r.driverId = d.driverId
        JOIN `{PROJECT_ID}.{DATASET}.races` rc   ON r.raceId  = rc.raceId
        WHERE rc.year >= 2022
          AND d.url IS NOT NULL AND d.url != ''
        ORDER BY d.surname
    """
    for row in client.query(recent_sql).result():
        _add(_extract_title(row.url), f"{row.forename} {row.surname} (current grid)")

    # 3. Top constructors by wins
    print("  Top constructors...")
    constructor_sql = f"""
        SELECT c.name, c.url, COUNT(*) AS wins
        FROM `{PROJECT_ID}.{DATASET}.results` r
        JOIN `{PROJECT_ID}.{DATASET}.constructors` c ON r.constructorId = c.constructorId
        WHERE r.position = 1
          AND c.url IS NOT NULL AND c.url != ''
        GROUP BY c.name, c.url
        ORDER BY wins DESC
        LIMIT 5
    """
    for row in client.query(constructor_sql).result():
        _add(_extract_title(row.url), f"{row.name} (constructor, {row.wins} wins)")

    return page_titles


def _chunk_text(text: str, title: str) -> list[dict]:
    """Split text into overlapping fixed-size chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end        = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "text":        chunk_text,
            "source":      title,
            "chunk_index": len(chunks),
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed texts in small batches with retry + rate limit handling.
    Free tier safe: batch size 3, 3s sleep between batches.
    """
    BATCH_SIZE     = 3
    all_embeddings = []
    total          = len(texts)

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        for attempt in range(5):
            try:
                result = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=batch,
                    task_type="retrieval_document",
                )
                all_embeddings.extend(result["embedding"])
                print(f"  Embedded {min(i + BATCH_SIZE, total)}/{total} chunks...")
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = (attempt + 1) * 15
                    print(f"  Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise e

        time.sleep(3)

    return np.array(all_embeddings, dtype="float32")


def _load_existing() -> tuple[list[dict], set[str]]:
    """Load existing chunks and return (chunks, set of already-scraped sources)."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            chunks = pickle.load(f)
        return chunks, set(c["source"] for c in chunks)
    return [], set()


def build_index(force_rebuild: bool = False):
    """
    Build or incrementally update the FAISS text index.

    force_rebuild=True  → wipe and rebuild from scratch
    force_rebuild=False → only scrape + embed pages not already indexed
    """
    if force_rebuild and os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
        os.remove(METADATA_PATH)
        print("Existing index wiped — rebuilding from scratch")

    existing_chunks, already_scraped = _load_existing()

    if already_scraped:
        print(f"Existing index: {len(already_scraped)} sources, "
              f"{len(existing_chunks)} chunks")
    else:
        print("No existing index — building from scratch")

    print("\nGetting page list from BigQuery...")
    all_titles = _get_pages_from_bigquery(top_n_drivers=20)

    new_titles = [t for t in all_titles if t not in already_scraped]
    print(f"\n{len(all_titles)} total pages | "
          f"{len(already_scraped)} already indexed | "
          f"{len(new_titles)} to scrape")

    if not new_titles:
        print("Index is up to date — nothing to do")
        return

    # Scrape new pages only
    new_chunks = []
    print(f"\nScraping {len(new_titles)} new pages...")
    for title in new_titles:
        print(f"  Fetching {title}...", end=" ", flush=True)
        text = _fetch_wikipedia(title)
        if not text:
            print("skipped")
            continue
        chunks = _chunk_text(text, title)
        new_chunks.extend(chunks)
        print(f"{len(chunks)} chunks")
        time.sleep(0.5)

    if not new_chunks:
        print("No new content to embed")
        return

    # Embed new chunks
    print(f"\nEmbedding {len(new_chunks)} new chunks...")
    texts      = [c["text"] for c in new_chunks]
    embeddings = _embed_batch(texts)
    faiss.normalize_L2(embeddings)

    # Merge into existing index or create fresh
    if os.path.exists(INDEX_PATH) and existing_chunks:
        index = faiss.read_index(INDEX_PATH)
        index.add(embeddings)
        all_chunks = existing_chunks + new_chunks
        print(f"Merged into existing index")
    else:
        index      = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        all_chunks = new_chunks

    # Save
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nIndex updated — {len(all_chunks)} total chunks "
          f"across {len(set(c['source'] for c in all_chunks))} sources")


def load_index():
    """Load FAISS text index from disk."""
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """
    Given a natural language question, return the top_k
    most relevant text chunks from driver/constructor bios.
    """
    if not os.path.exists(INDEX_PATH):
        build_index()

    index, metadata = load_index()

    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=[question],
        task_type="retrieval_query",
    )
    q_embedding = np.array(result["embedding"], dtype="float32")
    faiss.normalize_L2(q_embedding)

    scores, indices = index.search(q_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = metadata[idx].copy()
        chunk["relevance_score"] = round(float(score), 4)
        results.append(chunk)

    return results


def index_stats():
    """Print a summary of what's currently in the index."""
    chunks, sources = _load_existing()
    if not chunks:
        print("No index found")
        return
    print(f"Total chunks : {len(chunks)}")
    print(f"Unique sources: {len(sources)}")
    print()
    for s in sorted(sources):
        count = sum(1 for c in chunks if c["source"] == s)
        print(f"  {s}: {count} chunks")


if __name__ == "__main__":
    # Incrementally add any missing pages (won't re-scrape existing ones)
    build_index()

    # Show what's in the index
    print()
    index_stats()

    # Test retrieval
    print()
    test_questions = [
        "What made Senna so special in wet weather conditions?",
        "What is Hamilton's relationship with Mercedes?",
        "How did Verstappen develop as a young driver?",
        "Tell me about Oscar Piastri's rise in F1",
    ]
    for question in test_questions:
        print(f"Q: {question}")
        results = retrieve(question, top_k=2)
        for r in results:
            preview = r["text"][:100].replace("\n", " ")
            print(f"  → [{r['source']}] score:{r['relevance_score']}")
            print(f"     \"{preview}...\"")
        print()