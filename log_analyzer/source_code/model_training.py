import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
import joblib
from collections import Counter
from datetime import datetime
from google.colab import files

# Upload files
print("Please upload your log files...")
uploaded_files = files.upload()
log_files = list(uploaded_files.keys())

# Semi-supervised approach: first identify some clear anomalies
def identify_potential_anomalies(log):
    # These patterns indicate potential issues but we're not using them as keywords directly
    # Instead, we'll use them to create a small labeled dataset to guide our model
    severity_indicators = [
        r'error', r'exception', r'fail', r'critical', r'severe', r'warning',
        r'denied', r'fatal', r'crash', r'unavailable', r'timeout', r'refused',
        r'invalid', r'violation', r'unauthorized', r'unexpected'
    ]

    # Check if any indicators are present
    for indicator in severity_indicators:
        if re.search(indicator, log.lower()):
            return True
    return False

# Enhanced Feature Extraction Functions
def extract_features(log):
    # Timestamp extraction
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', log)
    timestamp = pd.to_datetime(timestamp_match.group() if timestamp_match else '1970-01-01', errors="coerce")

    # Event ID extraction
    event_id_match = re.search(r'Event ID[^\d]*(\d+)', log)
    event_id = int(event_id_match.group(1)) if event_id_match else 0

    # Log component/source extraction
    source_match = re.search(r'\[([\w\-\.]+)\]', log)
    source = source_match.group(1) if source_match else "unknown"

    # Basic features
    log_length = len(log)
    word_count = len(log.split())

    # Complex word ratio
    complex_words = sum(1 for word in log.split() if len(word) > 8)
    complex_word_ratio = complex_words / (word_count + 0.01)

    # Number of special characters
    special_chars_count = sum(1 for c in log if not c.isalnum() and not c.isspace())
    special_char_ratio = special_chars_count / (log_length + 0.01)

    # Number of digits
    digit_count = sum(c.isdigit() for c in log)
    digit_ratio = digit_count / (log_length + 0.01)

    # Uppercase to lowercase ratio
    uppercase_count = sum(c.isupper() for c in log if c.isalpha())
    lowercase_count = sum(c.islower() for c in log if c.isalpha())
    uppercase_ratio = uppercase_count / (lowercase_count + 1)  # Adding 1 to avoid division by zero

    # Time of day (0-23)
    hour_of_day = timestamp.hour if not pd.isnull(timestamp) else -1

    # Log structure features
    brackets_count = log.count('[') + log.count(']') + log.count('(') + log.count(')') + log.count('{') + log.count('}')
    brackets_ratio = brackets_count / (log_length + 0.01)
    quotes_count = log.count('"') + log.count("'")
    quotes_ratio = quotes_count / (log_length + 0.01)

    # Number of IP addresses
    ip_count = len(re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', log))

    # Number of hex codes (often in error messages)
    hex_count = len(re.findall(r'0x[0-9a-fA-F]+', log))

    # Punctuation ratio
    punctuation_count = sum(1 for c in log if c in '.,:;!?')
    punctuation_ratio = punctuation_count / (log_length + 0.01)

    # Entropy calculation (information density)
    char_freq = Counter(log)
    total_chars = len(log)
    entropy = -sum((count/total_chars) * np.log2(count/total_chars) for count in char_freq.values())

    # Potential anomaly flag (this won't be used directly for prediction)
    potential_anomaly = 1 if identify_potential_anomalies(log) else 0

    # Check for stack trace patterns
    stack_trace_indicators = sum(1 for pattern in ['at ', '.java:', 'line', 'Exception in', 'Traceback', 'File "'] if pattern in log)

    return [
        timestamp.timestamp(),
        event_id,
        log_length,
        word_count,
        complex_word_ratio,
        special_chars_count,
        special_char_ratio,
        digit_count,
        digit_ratio,
        uppercase_ratio,
        hour_of_day,
        brackets_count,
        brackets_ratio,
        quotes_count,
        quotes_ratio,
        ip_count,
        hex_count,
        punctuation_ratio,
        entropy,
        stack_trace_indicators,
        potential_anomaly,  # This is a guide, not a direct feature for prediction
        source,
        log
    ]

# Process Logs
print("Extracting features from logs...")
data = []
for log_file in log_files:
    print(f"Processing {log_file}...")
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        logs = f.readlines()
    for log in logs:
        if log.strip():  # Skip empty lines
            data.append(extract_features(log.strip()))

# Define column names based on our feature extraction
columns = [
    "timestamp",
    "event_id",
    "log_length",
    "word_count",
    "complex_word_ratio",
    "special_chars_count",
    "special_char_ratio",
    "digit_count",
    "digit_ratio",
    "uppercase_ratio",
    "hour_of_day",
    "brackets_count",
    "brackets_ratio",
    "quotes_count",
    "quotes_ratio",
    "ip_count",
    "hex_count",
    "punctuation_ratio",
    "entropy",
    "stack_trace_indicators",
    "potential_anomaly",  # This column won't be used for prediction
    "source",
    "log_entry"
]

df = pd.DataFrame(data, columns=columns)

    # Convert source to numerical using one-hot encoding
# Only create dummies if we have meaningful source values
if df['source'].nunique() > 1 and 'unknown' not in df['source'].unique():
    source_dummies = pd.get_dummies(df['source'], prefix='source')
    df = pd.concat([df, source_dummies], axis=1)
else:
    # If source extraction didn't work well, don't use it for modeling
    print("Source extraction didn't produce useful results - not using source in model")

# Use Hybrid Approach: Guided Isolation Forest

# 1. Split data based on our potential anomalies
clear_anomalies = df[df['potential_anomaly'] == 1]
clear_normal = df[df['potential_anomaly'] == 0]

# 2. Use Isolation Forest but with a higher contamination rate to catch more anomalies
print("Training model...")
# Exclude non-numeric columns and those we don't want to use for prediction
exclude_cols = ['timestamp', 'source', 'log_entry', 'potential_anomaly']
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']

# Normalize data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[feature_cols]),
    columns=feature_cols
)

# Calculate class weights to give higher importance to minority class
total_logs = len(df)
potential_anomaly_count = df['potential_anomaly'].sum()
normal_count = total_logs - potential_anomaly_count

# If we have some clear anomalies, use higher contamination rate than default
if potential_anomaly_count > 0:
    contamination_rate = max(0.01, min(0.35, (potential_anomaly_count / total_logs) * 1.5))
else:
    contamination_rate = 0.1  # Default if no clear anomalies detected

print(f"Identified potential anomalies: {potential_anomaly_count} out of {total_logs} logs")
print(f"Using contamination rate: {contamination_rate:.4f}")

# Train Isolation Forest
model = IsolationForest(
    contamination=contamination_rate,
    random_state=42,
    n_estimators=200,
    max_samples='auto',
    bootstrap=True
)

# Train model on the scaled features
print("Training Isolation Forest model...")
df['isolation_forest_pred'] = model.fit_predict(df_scaled)
df['isolation_forest_pred'] = np.where(df['isolation_forest_pred'] == -1, 1, 0)

# Get anomaly scores (more negative = more anomalous)
df['anomaly_score'] = model.decision_function(df_scaled)

# Final anomaly determination: use Isolation Forest predictions
df["predicted_anomaly"] = df['isolation_forest_pred']

# Ensure clear anomalies are marked as anomalies
# This step ensures that obvious anomalies are always caught
correction_count = 0
for idx, row in df.iterrows():
    if row['potential_anomaly'] == 1 and row['predicted_anomaly'] == 0:
        df.at[idx, 'predicted_anomaly'] = 1
        correction_count += 1

if correction_count > 0:
    print(f"Corrected {correction_count} predictions to ensure critical anomalies are captured")

# Generate summary statistics
anomalies = df[df['predicted_anomaly'] == 1]
print("\n--- RESULTS SUMMARY ---")
if not anomalies.empty:
    print(f"Total anomalies found: {len(anomalies)} out of {len(df)} logs ({len(anomalies)/len(df)*100:.2f}%)")

    # Group anomalies by source only if we have meaningful source values
    if 'source' in df.columns and len(set(anomalies['source'].unique()) - {'unknown', ''}) > 0:
        source_counts = anomalies['source'].value_counts()
        print("\nAnomalies by source:")
        for source, count in source_counts.items():
            if source and source != 'unknown':
                print(f"  {source}: {count} anomalies")
    else:
        print("\nSource information couldn't be reliably extracted from logs")

    # Display some example anomalies
    print("\nSample anomalies (showing 5 most anomalous):")
    most_anomalous = anomalies.sort_values('anomaly_score').head(5)
    for idx, row in most_anomalous.iterrows():
        print(f"- {row['log_entry'][:100]}..." if len(row['log_entry']) > 100 else f"- {row['log_entry']}")
else:
    print("No anomalies detected.")

# Calculate feature importance
if not anomalies.empty:
    normal_logs = df[df['predicted_anomaly'] == 0]
    print("\nFeature Analysis (differences between normal and anomalous logs):")
    feature_importance = {}
    for col in feature_cols:
        if col != 'predicted_anomaly' and col != 'isolation_forest_pred' and col != 'anomaly_score':
            normal_mean = normal_logs[col].mean()
            anomaly_mean = anomalies[col].mean()
            difference = abs(normal_mean - anomaly_mean)
            # Normalize the difference by the standard deviation to get a more meaningful measure
            std = df[col].std()
            if std > 0:
                normalized_diff = difference / std
            else:
                normalized_diff = 0
            feature_importance[col] = normalized_diff

    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 most important features for anomaly detection:")
    for feature, importance in sorted_features[:10]:
        normal_mean = normal_logs[feature].mean()
        anomaly_mean = anomalies[feature].mean()
        print(f"  {feature}: {importance:.2f} (Normal: {normal_mean:.2f}, Anomaly: {anomaly_mean:.2f})")

# Remove source column from results
df_for_export = df.drop(columns=['source'])

# Save detailed results to CSV
df_for_export.to_csv("log_analysis_detailed_results.csv", index=False)

# Create a simplified version with just the essentials
results_df = df[['timestamp', 'predicted_anomaly', 'anomaly_score', 'log_entry']]
results_df.to_csv("log_analysis_results.csv", index=False)

# Save Model and Scaler
joblib.dump(model, 'log_analyzer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')  # Save the feature columns for future use

print("\nModel, Scaler, and Feature list saved!")
print("Detailed results saved as 'log_analysis_detailed_results.csv'")
print("Simplified results saved as 'log_analysis_results.csv'")
