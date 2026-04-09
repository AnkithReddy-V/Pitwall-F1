import os
import json
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID")
DATASET    = os.getenv("BIGQUERY_DATASET")

INDEX_PATH    = "data/schema_index.faiss"
METADATA_PATH = "data/schema_metadata.pkl"

# ── Schema definitions ──────────────────────────────────────────────
# Each entry describes a table: what it contains, its key columns,
# and example questions it can answer. This is what gets embedded.

SCHEMAS = [
    {
        "table": "races",
        "description": "Every F1 race from 1950 to 2024. Contains race name, year, round number, circuit, date.",
        "columns": "raceId, year, round, circuitId, name, date",
        "example_questions": "Which year had the most races? What races were held at Monaco?"
    },
    {
        "table": "results",
        "description": "Race results for every driver in every race. Contains finishing position, grid position, points, fastest lap, DNF status.",
        "columns": "resultId, raceId, driverId, constructorId, grid, position, points, laps, fastestLapTime, statusId",
        "example_questions": "Who has the most wins? Which driver scores most points per race? Best grid to finish position improvement?"
    },
    {
        "table": "lap_times",
        "description": "Lap-by-lap times for every driver in every race from 1996 onwards. Contains lap number, position, time in milliseconds.",
        "columns": "raceId, driverId, lap, position, time, milliseconds",
        "example_questions": "Who has the most consistent lap times? Fastest average lap at Silverstone? Lap time progression through a race?"
    },
    {
        "table": "pit_stops",
        "description": "Every pit stop from 2011 onwards. Contains stop number, lap, duration in seconds and milliseconds.",
        "columns": "raceId, driverId, stop, lap, time, duration, milliseconds",
        "example_questions": "Which team has fastest average pit stop? Most pit stops in a race? Pit stop strategy analysis?"
    },
    {
        "table": "qualifying",
        "description": "Qualifying session results with Q1, Q2, Q3 times for each driver.",
        "columns": "qualifyId, raceId, driverId, constructorId, position, q1, q2, q3",
        "example_questions": "Best qualifying performance? Gap between pole and second? Qualifying vs race pace comparison?"
    },
    {
        "table": "drivers",
        "description": "Driver biographical information. Contains name, nationality, date of birth, Wikipedia URL.",
        "columns": "driverId, driverRef, code, forename, surname, dob, nationality",
        "example_questions": "How many British drivers? Driver nationality breakdown? Find driver by name?"
    },
    {
        "table": "constructors",
        "description": "Constructor (team) information. Contains team name, nationality, Wikipedia URL.",
        "columns": "constructorId, constructorRef, name, nationality",
        "example_questions": "How many Italian constructors? Team history? Find constructor by name?"
    },
    {
        "table": "circuits",
        "description": "Circuit information including location, country, GPS coordinates.",
        "columns": "circuitId, circuitRef, name, location, country, lat, lng",
        "example_questions": "Which circuits are in Europe? Highest altitude circuit? Street circuits vs permanent?"
    },
    {
        "table": "driver_standings",
        "description": "Championship standings after each race. Contains points, position, wins at each point in the season.",
        "columns": "driverStandingsId, raceId, driverId, points, position, wins",
        "example_questions": "Who led championship most races? Biggest points lead? Championship battles?"
    },
    {
        "table": "constructor_standings",
        "description": "Constructor championship standings after each race.",
        "columns": "constructorStandingsId, raceId, constructorId, points, position, wins",
        "example_questions": "Most dominant constructor season? Closest constructor battle?"
    },
    {
        "table": "status",
        "description": "Lookup table for race finish status codes. Maps statusId to description like Finished, Collision, Engine failure.",
        "columns": "statusId, status",
        "example_questions": "Most common DNF reason? Engine failures by team? Collision statistics?"
    },
    {
        "table": "glamour_index",
        "description": "Custom enrichment table with driver brand value, social following, luxury brand partnerships, and glamour index score.",
        "columns": "driverId, driver_ref, full_name, primary_brand_partner, brand_tier, social_following_M, est_market_value_M_USD, glamour_index",
        "example_questions": "Most commercially valuable driver? Highest glamour index? Overhyped drivers? Brand tier analysis?"
    },
]

def _schema_to_text(schema: dict) -> str:
    """Convert a schema dict to a single string for embedding."""
    return (
        f"Table: {schema['table']}. "
        f"{schema['description']} "
        f"Columns: {schema['columns']}. "
        f"Example questions: {schema['example_questions']}"
    )

def _embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using Gemini embedding model."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texts,
        task_type="retrieval_document",
    )
    return np.array(result["embedding"], dtype="float32")

def build_index():
    """Embed all schemas and save FAISS index to disk."""
    print("Building schema index...")
    texts = [_schema_to_text(s) for s in SCHEMAS]

    # Embed all schemas — one API call for the whole batch
    embeddings = _embed(texts)

    # Normalise vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build flat index (exact search — fine for 12 vectors)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(SCHEMAS, f)

    print(f"Index built — {index.ntotal} schemas indexed")

def load_index():
    """Load FAISS index from disk."""
    index    = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """
    Given a natural language question, return the top_k
    most relevant table schemas.
    """
    if not os.path.exists(INDEX_PATH):
        build_index()

    index, metadata = load_index()

    # Embed the question
    q_embedding = _embed([question])
    faiss.normalize_L2(q_embedding)

    # Search
    scores, indices = index.search(q_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        schema = metadata[idx].copy()
        schema["relevance_score"] = round(float(score), 4)
        results.append(schema)

    return results


if __name__ == "__main__":
    # Build the index
    build_index()

    # Test with sample questions
    test_questions = [
        "Who has the most pit stop errors?",
        "Which driver has the highest glamour index?",
        "What is the fastest lap time at Monaco?",
    ]

    print()
    for question in test_questions:
        print(f"Q: {question}")
        results = retrieve(question, top_k=2)
        for r in results:
            print(f"  → {r['table']} (score: {r['relevance_score']})")
        print()