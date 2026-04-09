import os
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID")
DATASET    = os.getenv("BIGQUERY_DATASET")

client = bigquery.Client(project=PROJECT_ID)

def load_glamour_index():
    csv_path = "data/glamour_index.csv"
    print("Loading glamour_index...", end=" ", flush=True)

    df = pd.read_csv(csv_path)
    table_id = f"{PROJECT_ID}.{DATASET}.glamour_index"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    table = client.get_table(table_id)
    print(f"done — {table.num_rows} rows loaded")

    # Verify with a quick query
    print("\nTop 5 by glamour index:\n")
    sql = f"""
        SELECT full_name, primary_brand_partner, 
               brand_tier, social_following_M, glamour_index
        FROM `{PROJECT_ID}.{DATASET}.glamour_index`
        ORDER BY glamour_index DESC
        LIMIT 5
    """
    query_job = client.query(sql)
    for row in query_job.result():
        print(f"  {row.full_name}: {row.glamour_index} "
              f"(brand tier {row.brand_tier}, "
              f"{row.social_following_M}M followers)")

if __name__ == "__main__":
    load_glamour_index()