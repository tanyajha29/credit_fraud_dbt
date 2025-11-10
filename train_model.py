import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
# --- UPDATED IMPORTS ---
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score # <--- ADDED accuracy_score
from datetime import datetime

# Tell MLflow where the server is (run `mlflow ui` first)
mlflow.set_tracking_uri("http://localhost:5000")

# --- !! CONFIGURE YOUR DATABASE HERE !! ---
DB_USER = "postgres"      # Your username
DB_PASS = "root"      # Your password
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "credit_fraud_dw" # Your database name
# ------------------------------------------

DB_SCHEMA = "analytics"
DB_TABLE = "stg_transactions" # This matches the dbt model filename

# --- Custom Transformers for Feature Engineering ---
# (These are unchanged)

class DatetimeTransformer(BaseEstimator, TransformerMixin):
    """Extracts the hour of the day from a timestamp."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expects a DataFrame column
        return X.apply(lambda col: col.dt.hour).values.reshape(-1, 1)

class AgeTransformer(BaseEstimator, TransformerMixin):
    """Calculates age from date of birth."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expects a DataFrame column
        today = datetime.now()
        # Ensure input is datetime before calculating days
        X_dt = pd.to_datetime(X.iloc[:,0]) # Convert Series to datetime if not already
        return X_dt.apply(lambda dob: (today - dob).days / 365.25).values.reshape(-1, 1)


class HaversineTransformer(BaseEstimator, TransformerMixin):
    """Calculates the Haversine distance between customer and merchant."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expects a DataFrame with specific column names
        R = 6371  # Earth radius in kilometers

        lat1 = np.radians(X['customer_latitude'])
        lon1 = np.radians(X['customer_longitude'])
        lat2 = np.radians(X['merchant_latitude'])
        lon2 = np.radians(X['merchant_longitude'])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * c
        return distance.values.reshape(-1, 1)

# --- End Custom Transformers ---


def get_data_from_warehouse():
    print("Connecting to data warehouse...")
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    query = f'SELECT * FROM {DB_SCHEMA}."{DB_TABLE}"' # Use quotes for dbt table name
    try:
        data = pd.read_sql(query, engine, parse_dates=['transaction_timestamp', 'date_of_birth'])
        print(f"Data loaded successfully: {len(data)} rows.")
        # Fill potential NaNs in coordinate columns before Haversine
        coord_cols = ['customer_latitude', 'customer_longitude', 'merchant_latitude', 'merchant_longitude']
        for col in coord_cols:
             if data[col].isnull().any():
                  print(f"Warning: Filling NaNs in {col} with 0.")
                  data[col].fillna(0, inplace=True) # Simple imputation for missing coordinates
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Did `dbt run` complete successfully and create the table '{DB_SCHEMA}.{DB_TABLE}'?")
        exit()


def train_model():
    df = get_data_from_warehouse()

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # --- Define Preprocessing ---
    # (This section is unchanged)
    numeric_features = ['amount', 'city_population']
    categorical_features = ['category', 'gender', 'job']
    datetime_features = ['transaction_timestamp'] # Single column name
    age_features = ['date_of_birth'] # Single column name
    distance_features = ['customer_latitude', 'customer_longitude', 'merchant_latitude', 'merchant_longitude']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    class AgeTransformerForCT(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X):
            today = datetime.now()
            X_dt = pd.to_datetime(X.iloc[:,0])
            return X_dt.apply(lambda dob: (today - dob).days / 365.25).values.reshape(-1, 1)

    class DatetimeTransformerForCT(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X):
            return X.iloc[:,0].dt.hour.values.reshape(-1, 1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('time', DatetimeTransformerForCT(), datetime_features),
            ('age', AgeTransformerForCT(), age_features),
            ('dist', HaversineTransformer(), distance_features)
        ],
        remainder='drop'
    )

    # --- Create Full Model Pipeline ---
    # (This is unchanged)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_estimators=100,
            n_jobs=-1
        ))
    ])

    # --- Train and Log with MLflow ---
    with mlflow.start_run() as run:
        print("Training model...")
        model_pipeline.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = model_pipeline.predict(X_test)
        y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

        # --- UPDATED METRIC CALCULATIONS ---
        accuracy = accuracy_score(y_test, y_pred) # <--- ADDED
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # --- UPDATED PRINT STATEMENTS ---
        print("\n--- Model Evaluation Metrics ---")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Recall:    {recall:.4f}  (Ability to catch fraud)")
        print(f"Precision: {precision:.4f} (Correctness when predicting fraud)")
        print(f"F1-Score:  {f1:.4f}   (Balance of Recall & Precision)")
        print("------------------------------\n")
        # ---------------------------------

        # --- Log Probabilities for Analysis ---
        # (This section is unchanged)
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Fraud_Probability': y_pred_proba
        })
        actual_fraud_results = results_df[results_df['Actual'] == 1]
        print("\n--- Probabilities for Actual Fraud Cases (Sample) ---")
        print(actual_fraud_results.head())
        # Add check for empty dataframe to avoid .mean() error if no fraud in test set
        avg_prob_fraud = actual_fraud_results['Fraud_Probability'].mean() if not actual_fraud_results.empty else 0.0
        print(f"Average probability for actual fraud: {avg_prob_fraud:.4f}")


        actual_non_fraud_results = results_df[results_df['Actual'] == 0]
        print("\n--- Probabilities for Actual Non-Fraud Cases (Sample) ---")
        print(actual_non_fraud_results.head())
        avg_prob_non_fraud = actual_non_fraud_results['Fraud_Probability'].mean() if not actual_non_fraud_results.empty else 0.0
        print(f"Average probability for actual non-fraud: {avg_prob_non_fraud:.4f}")
        # --- End Probability Logging ---


        # --- UPDATED MLFLOW LOGGING ---
        mlflow.log_metric("accuracy", accuracy) # <--- ADDED
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        # -----------------------------

        # Log the entire pipeline
        mlflow.sklearn.log_model(model_pipeline, "fraud_detector_pipeline")

        run_id = run.info.run_id
        print(f"\nModel training complete.")
        print(f"Run ID: {run_id}")
        print(f"Model saved in MLflow. Access UI at http://localhost:5000")

        return run_id

if __name__ == "__main__":
    train_model()