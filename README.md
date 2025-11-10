# End-to-End Credit Card Fraud Detection System

This project is a complete, end-to-end system for detecting credit card fraud using a professional data stack. It uses a raw, simulated dataset with real-world features, which are processed using a full data pipeline.

## Project Architecture



The workflow is as follows:
1.  **Load:** A Python script (`load_raw_data.py`) loads the raw `fraudTrain.csv` into a PostgreSQL database in a `raw_data` schema.
2.  **Transform:** `dbt` is used to transform the raw data. It cleans columns, casts datatypes, and creates a clean `analytics.stg_transactions` table.
3.  **Train:** A Python script (`train.py`) reads from the dbt table. It uses a `scikit-learn` `Pipeline` to perform all feature engineering, including:
    * Calculating customer age from `date_of_birth`.
    * Calculating the Haversine distance between customer and merchant.
    * Extracting the hour from the transaction timestamp.
    * One-hot encoding categorical features.
    * Scaling numerical features.
    * This full pipeline is logged to `MLflow`.
4.  **Serve:** A `FastAPI` application (`api.py`) loads the entire pipeline from MLflow and serves it at a `/predict` endpoint.
5.  **Present:** A `Streamlit` dashboard (`dashboard.py`) provides a UI to enter new transaction data, which it sends to the FastAPI backend for a live prediction.

## Tech Stack

* **Data Warehouse:** PostgreSQL
* **Data Transformation:** dbt
* **ML Experiment Tracking:** MLflow
* **Feature Engineering/Model:** Scikit-learn, Pandas
* **Backend API:** FastAPI
* **Frontend Dashboard:** Streamlit

## How to Run

You will need 4 terminals open.

1.  **Prerequisites:**
    * Install Python 3.9+, PostgreSQL.
    * Create a database in Postgres named `fraud_db`.
    * Download `fraudTrain.csv` into a `/data` folder.

2.  **Install & Setup:**
    ```bash
    git clone [your-repo-link]
    cd fraud_detection_system
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt 
    # (You should create this file: pip freeze > requirements.txt)
    ```

3.  **Phase 1: Load Data & Run dbt**
    * Update your database credentials in `load_raw_data.py`.
    * Run: `python load_raw_data.py`
    * Navigate to `fraud_dbt_project/` and update `profiles.yml` (in `~/.dbt/`) with your DB details.
    * Run: `cd fraud_dbt_project`
    * Run: `dbt run`
    * Go back to the main folder: `cd ..`

4.  **Terminal 1: MLflow UI**
    ```bash
    mlflow ui
    ```

5.  **Terminal 2: Model Training**
    * Run: `python train.py`
    * Wait for it to finish. Copy the **Run ID** it prints.

6.  **Terminal 3: Backend API**
    * Paste the **Run ID** into the `MLFLOW_RUN_ID` variable in `api.py`.
    * Run: `python api.py`

7.  **Terminal 4: Frontend Dashboard**
    ```bash
    streamlit run dashboard.py
    ```
    
8.  **View:** Open `http://localhost:8501` in your browser.