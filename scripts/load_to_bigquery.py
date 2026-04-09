import os
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID")
DATASET    = os.getenv("BIGQUERY_DATASET")
DATA_DIR   = "data/raw"

client = bigquery.Client(project=PROJECT_ID)

TABLES = [
    "races",
    "results",
    "lap_times",
    "pit_stops",
    "qualifying",
    "drivers",
    "constructors",
    "circuits",
    "driver_standings",
    "constructor_standings",
    "status",
]

def load_table(table_name: str):
    csv_path = os.path.join(DATA_DIR, f"{table_name}.csv")

    if not os.path.exists(csv_path):
        print(f"  SKIP  {table_name} — file not found")
        return

    print(f"  Loading {table_name}...", end=" ", flush=True)
    df = pd.read_csv(csv_path, na_values=["\\N"])

    table_id = f"{PROJECT_ID}.{DATASET}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    table = client.get_table(table_id)
    print(f"done — {table.num_rows:,} rows loaded")

if __name__ == "__main__":
    print(f"\nLoading F1 data into {PROJECT_ID}.{DATASET}\n")
    for table in TABLES:
        load_table(table)
    print("\nAll tables loaded successfully.")