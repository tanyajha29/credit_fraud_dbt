import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_evaluation_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculates and prints standard classification metrics.
    Also prints probability analysis for fraud/non-fraud classes.
    """
    # --- Calculate Standard Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # --- Print Standard Metrics ---
    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Recall:    {recall:.4f}  (Ability to catch fraud)")
    print(f"Precision: {precision:.4f} (Correctness when predicting fraud)")
    print(f"F1-Score:  {f1:.4f}   (Balance of Recall & Precision)")
    print("------------------------------\n")

    # --- Print Probability Analysis ---
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Fraud_Probability': y_pred_proba
    })

    # Filter for actual fraud and non-fraud cases
    actual_fraud_results = results_df[results_df['Actual'] == 1]
    actual_non_fraud_results = results_df[results_df['Actual'] == 0]

    # Calculate average probabilities if fraud cases exist
    avg_prob_fraud = actual_fraud_results['Fraud_Probability'].mean() if not actual_fraud_results.empty else 0
    avg_prob_non_fraud = actual_non_fraud_results['Fraud_Probability'].mean() if not actual_non_fraud_results.empty else 0

    print("--- Probabilities for Actual Fraud Cases (Sample) ---")
    print(actual_fraud_results.head())
    print(f"Average probability for actual fraud: {avg_prob_fraud:.4f}")

    print("\n--- Probabilities for Actual Non-Fraud Cases (Sample) ---")
    print(actual_non_fraud_results.head())
    print(f"Average probability for actual non-fraud: {avg_prob_non_fraud:.4f}")
    print("--------------------------------------------------\n")

    # Return the metrics if needed elsewhere (e.g., for logging)
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1
    }