import pandas as pd
import numpy as np
import re
import joblib
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from datetime import datetime
from google.colab import files

# Upload new log files for anomaly detection
print("Please upload your log files...")
uploaded_files = files.upload()
log_files = list(uploaded_files.keys())

# Load previously saved model, scaler, and feature columns
print("Loading saved model, scaler, and feature columns...")
model = joblib.load('log_analyzer_model.pkl')  # Load the trained Isolation Forest model
scaler = joblib.load('scaler.pkl')  # Load the MinMaxScaler
feature_cols = joblib.load('feature_cols.pkl')  # Load the list of feature columns

# Enhanced Feature Extraction Functions (same as before)
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

    # Stack trace indicators (useful for detecting errors)
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
        source,
        log
    ]

# Process the new logs and extract features
print("Extracting features from new logs...")
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
    "source",
    "log_entry"
]

df = pd.DataFrame(data, columns=columns)

# Exclude non-numeric columns for prediction
exclude_cols = ['timestamp', 'source', 'log_entry']
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'object']

# Normalize the data using the same scaler
df_scaled = pd.DataFrame(
    scaler.transform(df[feature_cols]),
    columns=feature_cols
)

# Predict anomalies using the loaded model
print("Predicting anomalies using the pre-trained model...")
df['isolation_forest_pred'] = model.predict(df_scaled)
df['isolation_forest_pred'] = np.where(df['isolation_forest_pred'] == -1, 1, 0)  # Convert -1 to 1 for anomalies

# Get anomaly scores
df['anomaly_score'] = model.decision_function(df_scaled)

# Show results
print("\n--- RESULTS SUMMARY ---")
anomalies = df[df['isolation_forest_pred'] == 1]
if not anomalies.empty:
    print(f"Total anomalies found: {len(anomalies)} out of {len(df)} logs ({len(anomalies)/len(df)*100:.2f}%)")

    # Display some example anomalies
    print("\nSample anomalies (showing 5 most anomalous):")
    most_anomalous = anomalies.sort_values('anomaly_score').head(5)
    for idx, row in most_anomalous.iterrows():
        print(f"- {row['log_entry'][:100]}..." if len(row['log_entry']) > 100 else f"- {row['log_entry']}")
else:
    print("No anomalies detected.")

# Save the results to a CSV file
results_df = df[['timestamp', 'isolation_forest_pred', 'anomaly_score', 'log_entry']]
results_df.to_csv("new_log_analysis_results.csv", index=False)

print("\nAnomaly detection completed! Results saved as 'new_log_analysis_results.csv'.")
