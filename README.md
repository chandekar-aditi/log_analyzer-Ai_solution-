# log_analyzer-Ai_solution-
# Overview
The Log Analyzer is a powerful tool that automates the detection of anomalies in system logs. It uses a hybrid approach by combining traditional anomaly detection methods like Isolation Forest with guided anomaly indicators to improve accuracy. The tool processes logs, extracts features, and identifies anomalies, which are then analyzed using a trained model to generate potential solutions.
This project helps streamline the log analysis process, automatically identifying issues, providing potential solutions, and saving the results for further review.

# Key Features
**1.Hybrid Anomaly Detection:** Combines traditional anomaly detection algorithms (Isolation Forest) with manually identified potential anomalies to improve the detection process.

**2.Feature Extraction:** Extracts various features from logs such as timestamps, event IDs, word count, entropy, IP addresses, special characters, and more.

**3.Potential Anomaly Identification:** Uses severity indicators (e.g., "error", "exception", "critical") to create a labeled dataset for the model.

**4.Efficient Processing:** Processes log files in bulk, extracts relevant features, and detects anomalies in a scalable way.

**5.Model Training:** Trains the model using a MinMax scaling approach to ensure uniform feature importance and higher accuracy.

# Model & Anomaly Detection Approach
**1.Hybrid Model:** The system uses a hybrid anomaly detection approach, combining a traditional unsupervised method (Isolation Forest) with manual indicators (e.g., error, exception, etc.) to identify potential anomalies.

**2.Isolation Forest:** This algorithm works by isolating observations based on their feature values, providing a score for each observation, where more negative scores indicate higher likelihoods of being an anomaly.

**3.Contamination Rate:** The contamination rate is dynamically calculated based on the detected potential anomalies in the dataset. This helps the model tune itself to detect more anomalies in cases with rare events.

# Performance & Efficiency
**1.Scalable:** Efficiently processes large log files by extracting relevant features and using scalable models like Isolation Forest.

**2.Dynamic Adjustments:** The contamination rate adjusts automatically based on the dataset, ensuring a balance between false positives and false negatives.

**3.Feature Importance Analysis:** After anomaly detection, the script analyzes which features are most important in distinguishing between normal and anomalous logs.

# Complete Workflow Overview 
  ## Step 1: Log Analysis & Anomaly Detection
  This script performs the initial log analysis, feature extraction, and anomaly detection using 
  the Isolation Forest model.<br/>
  **Key Actions:** <br/>
     **1. Upload Log Files:** Users upload their log files for analysis.<br/>
     **2. Feature Extraction:** For each log entry, features are extracted, including<br/>
        * Timestamp<br/>
        * Event ID<br/>
        * Log Length<br/>
        * Special character count<br/>
        * Complexity ratio<br/>
        * Entropy (information density)<br/>
        * Potential anomaly flags (based on keywords like error, fail, etc.)<br/>

   **3. Isolation Forest Model:** The features are scaled, and the Isolation Forest model is 
    trained to identify anomalies. Anomaly scores and predictions are stored for further 
    analysis.

   **4. Results Summary:** The script generates a summary of anomalies and their 
    characteristics, providing insights into which log entries were flagged as anomalous.

**Outputs:**

**Log Analysis CSV:** A detailed CSV file with all log features, including anomaly predictions (log_analysis_detailed_results.csv).

**Simplified Results CSV:** A CSV with essential anomaly data (log_analysis_results.csv).

**Model and Scaler:** The trained Isolation Forest model and scaler are saved for future predictions.

## Step 2: Clustering Anomalies
This script takes the detected anomalies from the previous step and groups them into clusters using DBSCAN clustering.

**Key Actions**<br/>
**1. Load Anomalous Logs:** The script loads the log analysis results and extracts entries flagged as anomalies.

**2. Feature Extraction for Clustering:** The TF-IDF vectoriser is used to convert the log text entries into numerical features for clustering.

**3. DBSCAN Clustering:** The DBSCAN algorithm is used to group similar anomalous logs based on cosine similarity of their TF-IDF features.

**4. AI Prompt Generation:** For each cluster, an AI prompt is generated to suggest possible solutions for the issues identified in the cluster.

**Outputs:**<br/>
**Clustered Anomalies CSV:** A CSV file with clustering information and AI-generated prompts for each anomaly cluster (anomaly_clusters_with_prompts.csv).

## Step 3: Generate AI Solutions
This script utilizes the LLaMA 2 (Meta AI) model to generate solutions for each cluster of anomalies.

**Key Actions:**<br/>
**1.Load LLaMA 2 Model:** The script loads the LLaMA 2 (7B) model with disk offloading for efficient memory usage.

**2.Generate Solutions:** For each AI prompt, the model generates a possible solution, providing recommendations or steps to address the anomalies in the log data.

**3.Save Solutions:** The generated solutions are saved into a text file for further review and implementation.

**Outputs:**

**AI Solutions File:** A text file with the generated solutions for each cluster of anomalies (sample_anomaly_solutions.txt).

## Notes:
 1. Make sure to upload log files when running the script.

2. Ensure the LLaMA 2 model is accessible and GPU resources are available for efficient processing.

# File Descriptions:
### 1. log_analysis_detailed_results.csv
This file contains detailed information about each log entry, including various features and anomaly predictions.

**Columns:**

  1.timestamp: The time when the log entry was created.

  2.event_id: A unique identifier for the event recorded in the log.

  3.log_entry: The actual log message.

  4.log_length: The length (number of characters) of the log entry.

  5.special_char_count: The number of special characters (like punctuation) in the log.

  6.complexity_ratio: A measure of the complexity of the log entry (could be based on word-to- 
    character ratio, etc.).

  7.entropy: The information density of the log entry (higher entropy means more complex).

  8.keyword_error: A boolean indicating if the log contains a keyword like "error" or "fail".

  9.anomaly_score: The score assigned by the Isolation Forest model, indicating how likely the 
  log is an anomaly.

 10.isolation_forest_pred: The prediction of the Isolation Forest model, where 1 indicates an 
  anomaly, and 0 indicates normal.

### 2. log_analysis_results.csv
This file provides a simplified summary of the detected anomalies from the detailed log analysis.

Columns:

timestamp: The time when the log entry was created.

log_entry: The actual log message.

isolation_forest_pred: The prediction of the Isolation Forest model (1 for anomaly, 0 for normal).

anomaly_score: The anomaly score from the Isolation Forest model.

3. anomaly_clusters_with_prompts.csv
This file contains clustered anomalies, with a prompt generated for each cluster to provide AI-based suggestions for solutions.

Columns:

cluster_label: The label assigned to the cluster by DBSCAN (each unique label represents a different cluster).

log_count: The number of log entries in this cluster.

example_logs: A few sample log entries from this cluster (up to 5).

ai_prompt: The AI-generated prompt that summarizes the cluster and asks for a solution. This will be used to generate suggestions from the LLaMA 2 model.

4. sample_anomaly_solutions.txt
This file contains the AI-generated solutions for each anomaly cluster. It provides potential solutions or actions to take for the detected issues.

Content:

Each solution corresponds to a cluster, providing the AI's suggested actions or resolutions for the anomalies identified in the logs. Each solution is separated by two new lines for readability.


