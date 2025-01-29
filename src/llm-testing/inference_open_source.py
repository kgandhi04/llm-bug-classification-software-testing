#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from prompt import classification_prompts
# Define available Ollama models
open_source_models = [
    "mistral",
    # "gemma",
    # "llava",
    # "llama3.1"
]

# Load dataset
df = pd.read_csv("test.csv")  # Ensure the correct filename
df["actual_label"] = df["labels"]  # Assuming there is a column with the true labels

# Select 100 rows for each label type
label_types = df["actual_label"].unique()
filtered_dfs = []

for label in label_types:
    filtered_df = df[df["actual_label"] == label].head(5)
    filtered_dfs.append(filtered_df)

df = pd.concat(filtered_dfs, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Function to perform inference using Ollama API
def ollama_inference(model_name, prompt):
    url = "http://localhost:11434/api/generate"  # Ollama default API endpoint

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # Set to True for streaming responses
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received")
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

# Function to classify text and extract the integer result
def classify_text(input_text, prompt, model_name, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = ollama_inference(model_name, input_text)
            # Extract integer using regex
            match = re.search(r'\b[0-3]\b', response)
            if match:
                return int(match.group(0))
            else:
                raise ValueError("No valid integer (0-3) found in response: ",response)
        except ValueError as ve:
            print(f"Invalid response: {ve}")
            return None
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for model {model_name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"Skipping model {model_name} after {max_retries} failed attempts.")
                return None

# Prepare to store evaluation results
results = []

# Running classification and storing results
for model in open_source_models:
    for i, prompt in enumerate(classification_prompts):
        col_name = f"classification_{model}_prompt{i+1}"
        try:
            df[col_name] = df["text"].apply(lambda x: classify_text(x, prompt, model))

            # Evaluate performance
            predicted_labels = df[col_name].dropna().astype(int)
            actual_labels = df["actual_label"].dropna().astype(int)

            min_length = min(len(predicted_labels), len(actual_labels))
            predicted_labels = predicted_labels[:min_length]
            actual_labels = actual_labels[:min_length]

            # Compute evaluation metrics
            conf_matrix = confusion_matrix(actual_labels, predicted_labels)
            accuracy = accuracy_score(actual_labels, predicted_labels)
            precision = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
            recall = recall_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
            f1 = f1_score(actual_labels, predicted_labels, average='weighted', zero_division=0)

            # Store evaluation results
            results.append({
                "Model": model,
                "Prompt": f"Prompt {i+1}",
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4),
            })

            # Save confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=np.unique(actual_labels), yticklabels=np.unique(actual_labels))
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model} - Prompt {i+1}\nAccuracy: {accuracy:.4f}")
            plt.savefig(f"confusion_matrix_{model}_prompt{i+1}.png")
            plt.close()

            # Generate CSV for incorrect classifications
            incorrect_classifications = df[df[col_name] != df["actual_label"]]
            incorrect_classifications[["text", "actual_label", col_name]].to_csv(
                f"incorrect_classifications_{model}_prompt{i+1}.csv", index=False
            )

            print(f"Metrics for {model} - Prompt {i+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        except Exception as e:
            print(f"Skipping classification for model {model} with prompt {i+1} due to error: {e}")

# Convert results to dataframe
results_df = pd.DataFrame(results)

# Save the metrics in a CSV file
results_df.to_csv("classification_metrics_open_source.csv", index=False)

# Display tabular format
print("Final Performance Metrics:")
print(results_df)

# Save the final classified results
df.to_csv("classified_results_open_source.csv", index=False)
print("Classification complete. Results, performance plots, and incorrect classifications saved.")
#%%
