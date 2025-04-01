import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

# Load detected anomalies from the previous analysis
print("Loading anomaly data...")
df = pd.read_csv('new_log_analysis_results.csv')
anomalies = df[df['isolation_forest_pred'] == 1]

if anomalies.empty:
    print("No anomalies found. Exiting...")
    exit()

# Extract logs and create a feature matrix using TF-IDF for clustering
print("Extracting log features with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(anomalies['log_entry'])

# Perform clustering using DBSCAN
print("Clustering similar anomalies...")
dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
anomalies['cluster'] = dbscan.fit_predict(X)

# Generate AI prompts for each cluster
print("Generating AI prompts for clusters...")
def generate_prompt(logs):
    example_logs = '\n'.join(logs[:3])
    return f"Analyze the following similar log errors and suggest solutions:\n{example_logs}\n..."

# Create a summary for each cluster
clustered_logs = []
for cluster_label in set(anomalies['cluster']):
    cluster_data = anomalies[anomalies['cluster'] == cluster_label]
    logs = cluster_data['log_entry'].tolist()
    prompt = generate_prompt(logs)
    clustered_logs.append({
        'cluster_label': cluster_label,
        'log_count': len(logs),
        'example_logs': '\n'.join(logs[:5]),
        'ai_prompt': prompt
    })

# Save results to CSV
print("Saving results to 'anomaly_clusters_with_prompts.csv'...")
pd.DataFrame(clustered_logs).to_csv('anomaly_clusters_with_prompts.csv', index=False)

print("✅ Anomaly clustering and AI prompt generation completed!")
