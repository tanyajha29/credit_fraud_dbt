import pandas as pd
from sqlalchemy import create_engine, text
import os

# --- !! CONFIGURE YOUR DATABASE HERE !! ---
DB_USER = "postgres"      # Replace with your PostgreSQL username
DB_PASS = "root"  # Replace with your PostgreSQL password
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "credit_fraud_dw"
# ------------------------------------------

RAW_SCHEMA_NAME = "raw_data"
RAW_TABLE_NAME = "raw_simulated_transactions" # Changed table name for clarity
CSV_PATH = os.path.join("data", "fraudTrain.csv") # Path to the Kartik dataset CSV

def load_data():
    print("Connecting to database...")
    try:
        engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

        with engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA_NAME}"))
            conn.commit() # Make sure schema is committed
            print(f"Schema '{RAW_SCHEMA_NAME}' ensured.")

        print(f"Reading CSV from {CSV_PATH}...")
        # Read the CSV in chunks for memory efficiency
        chunk_iter = pd.read_csv(CSV_PATH, chunksize=50000, index_col=0) # index_col=0 for this dataset

        first_chunk = True
        for i, chunk in enumerate(chunk_iter):
            print(f"Loading chunk {i+1}...")

            # Clean column names (lowercase, remove spaces)
            chunk.columns = [col.lower().replace(' ', '_') for col in chunk.columns]

            # Convert date columns to datetime objects
            chunk['trans_date_trans_time'] = pd.to_datetime(chunk['trans_date_trans_time'])
            chunk['dob'] = pd.to_datetime(chunk['dob'])

            # Use if_exists='replace' for the first chunk, 'append' for the rest
            write_mode = "replace" if first_chunk else "append"

            chunk.to_sql(
                RAW_TABLE_NAME,
                engine,
                schema=RAW_SCHEMA_NAME,
                if_exists=write_mode,
                index=False
            )
            first_chunk = False # Subsequent chunks should append
        print("Data loading complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check credentials, CSV path, and PostgreSQL status.")

if __name__ == "__main__":
    load_data()