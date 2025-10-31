import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np    
import json         
from evidently import Report   
from evidently.presets import DataDriftPreset 

from datetime import datetime

# Create timestamp for the run name
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

run_name = f"random-forest-v2-drifted-{timestamp}"

# Set the tracking URI to connnect to local Mlflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

experiment_name = "iris_classification_drift"


def delete_experiment_by_name(experiment_name: str):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    if exp is None:
        print(f"⚠️ No active or deleted experiment found with name '{experiment_name}'. Nothing to delete.")
        return
    
    exp_id = exp.experiment_id
    print(f"🗑️ Deleting experiment '{experiment_name}' (ID: {exp_id})")

    try:
        client.delete_experiment(exp_id)
        print(f"✅ Experiment '{experiment_name}' marked as deleted.")
    except Exception as e:
        print(f"⚠️ Could not delete experiment: {e}")

# --- Vor MLflow set_experiment() aufrufen ---
delete_experiment_by_name("iris_classification_drift")

# 3️⃣ Jetzt neu erstellen
mlflow.set_experiment(experiment_name)
print(f"✓ Experiment '{experiment_name}' recreated cleanly.")
print(f"✓ Experiment '{experiment_name}' is ready")


# Load data and prep
iris_data = load_iris(as_frame=True)
df = iris_data.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Start MLflow run
with mlflow.start_run(run_name="baseline_model") as run:
    
    # Enable autologging
    mlflow.sklearn.autolog()

    # Define hyperparamters
    n_estimators = 100
    max_depth = 5
    random_state = 42

    # Log parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("dataset", "iris")

    # Train model
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)   

    # Log the model using sklearn flavor
    mlflow.sklearn.log_model(sk_model=model, name="random_forest_model",registered_model_name="IrisRandomForestClassifierModel")    
    
    # Store run_id for later use
    run_id_v1 = run.info.run_id

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Run ID: {run_id_v1}")
    print(f"\nMetrics:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1 Score:  {f1:.4f}")
    print("\n✓ Model logged to MLflow")
    print(f"\n👉 View this run in the UI: http://127.0.0.1:5000")



# ============================================================================
# STEP 2: DRIFT DETECTION
# ============================================================================
X_drifted = X_test.copy()
X_drifted["sepal length (cm)"] += np.random.normal(loc=2.0, scale=0.3, size=len(X_drifted))

# Create and run drift report (UPDATED for v0.7.15)
report = Report(metrics=[DataDriftPreset()])

snapshot = report.run(current_data=X_drifted, reference_data=X_train)  # <-- UPDATED for v0.7.x Only way it works!
snapshot.save_html("drift_report.html")  

# Log drift report to same MLflow run
mlflow.log_artifact("drift_report.html", "drift_reports")
print("✓ Drift report saved to 'drift_report.html' and logged to MLflow")

# Test model performance on drifted data
print("\nStep 6: Evaluating model on drifted data...")
y_pred_drifted = model.predict(X_drifted)
accuracy_drifted = accuracy_score(y_test, y_pred_drifted)

print(f"Model accuracy on original data: {accuracy:.4f}")
print(f"Model accuracy on drifted data:  {accuracy_drifted:.4f}")
print(f"Performance change: {accuracy_drifted - accuracy:+.4f}")

# Log drifted performance metrics
mlflow.log_metric("accuracy_drifted", accuracy_drifted)
mlflow.log_metric("performance_change", accuracy_drifted - accuracy)

# Add this CLI output section
print("\n" + "="*50)
print("DRIFT DETECTION RESULTS")
print("="*50)

# Convert snapshot to JSON string and parse
json_str = snapshot.json()
drift_results = json.loads(json_str)
drift_metrics = drift_results.get("metrics", [])

# Parse the metrics
drifted_count = 0
total_features = 0
feature_drifts = []

for metric in drift_metrics:
    metric_id = metric.get("metric_id", "")
    
    if "DriftedColumnsCount" in metric_id:
        # This is the summary metric
        drifted_count = int(metric["value"]["count"])
        drift_share = metric["value"]["share"]
        total_features = int(drifted_count / drift_share) if drift_share > 0 else 0
        
    elif "ValueDrift" in metric_id:
        # This is a feature-level drift metric
        feature_name = metric_id.split("column=")[1].replace(")", "")
        drift_value = metric["value"]
        # Typically, low p-value (< 0.05) means drift detected
        is_drifted = drift_value < 0.05
        feature_drifts.append({
            "feature": feature_name,
            "drift_value": drift_value,
            "drifted": is_drifted
        })

# Print results
print(f"📊 DRIFT SUMMARY:")
print(f"   Drifted Features: {drifted_count}/{total_features} ({drift_share:.1%})")
print(f"   Dataset Drift: {'🚨 DETECTED' if drifted_count > 0 else '✅ No drift'}")

print(f"\n🔍 FEATURE-LEVEL ANALYSIS:")
for feature in feature_drifts:
    status = "🚨 DRIFT" if feature["drifted"] else "✅ No drift"
    print(f"   - {feature['feature']:20} {status} (p-value: {feature['drift_value']:.6f})")

    print(f"\n📝 INTERPRETATION:")
    print(f"   • p-value < 0.05: Significant drift detected")
    print(f"   • p-value >= 0.05: No significant drift")
    print(f"   • {drifted_count} feature(s) showing statistical drift")

    print("✓ Drift report saved to 'drift_report.html' and logged to MLflow")
    print(f"👉 View detailed report: http://127.0.0.1:5000")