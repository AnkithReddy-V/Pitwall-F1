import os
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID")
DATASET    = os.getenv("BIGQUERY_DATASET")

client = bigquery.Client(project=PROJECT_ID)

def run_query(sql: str) -> list[dict]:
    """Run a SQL query and return results as a list of dicts."""
    query_job = client.query(sql)
    results   = query_job.result()
    return [dict(row) for row in results]


if __name__ == "__main__":
    sql = f"""
        SELECT
            d.forename,
            d.surname,
            COUNT(*) AS wins
        FROM `{PROJECT_ID}.{DATASET}.results` r
        JOIN `{PROJECT_ID}.{DATASET}.drivers` d
            ON r.driverId = d.driverId
        WHERE r.position = 1
        GROUP BY d.forename, d.surname
        ORDER BY wins DESC
        LIMIT 5
    """
    print("\nTop 5 F1 drivers by race wins:\n")
    rows = run_query(sql)
    for row in rows:
        print(f"  {row['forename']} {row['surname']}: {row['wins']} wins")