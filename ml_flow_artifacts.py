import mlflow
from mlflow.tracking import MlflowClient

# Ensure the tracking URI is set correctly
mlflow.set_tracking_uri("http://localhost:5000") # Or your local file path

# Use the Run ID you want to inspect
run_id = "7576ee8e910843d59d1c65893f734df6" # Replace with your actual run ID

client = MlflowClient()

# --- Get Run Details ---
try:
    run = client.get_run(run_id)

    print("--- Run Info ---")
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Status: {run.info.status}")
    print(f"Start Time: {run.info.start_time}") # Timestamp in milliseconds
    print(f"End Time: {run.info.end_time}")     # Timestamp in milliseconds

    print("\n--- Metrics ---")
    print(run.data.metrics)

    print("\n--- Parameters ---")
    print(run.data.params)

    print("\n--- Tags ---")
    print(run.data.tags)

    # --- List Artifacts ---
    artifacts = client.list_artifacts(run_id)
    print("\n--- Artifacts ---")
    for artifact in artifacts:
        print(f"Path: {artifact.path}, Is Dir: {artifact.is_dir}, Size: {artifact.file_size}")

    # --- Download Artifacts (Optional) ---
    # Example: Download the entire saved model folder
    # local_download_path = "./downloaded_model"
    # client.download_artifacts(run_id, "fraud_detector_pipeline", local_download_path)
    # print(f"\nDownloaded 'fraud_detector_pipeline' artifact to {local_download_path}")

except Exception as e:
    print(f"Error fetching run {run_id}: {e}")