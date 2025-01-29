# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
from prompt import classification_prompts
import time
import re

# API Configuration
org_id = os.getenv("OPENAI_ORG_ID")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, organization=org_id)

# Load data
df = pd.read_csv("test_predictions_data.csv")  
df["actual_label"] = df["labels"]  
df["text"]=df["text"].apply(lambda x : x.replace("TITLE","").replace("BODY",""))

# Select 10 rows for each label type and combine into a single dataframe
label_types = df["actual_label"].unique()  # Get unique label values
filtered_dfs = []

for label in label_types:
    filtered_df = df[df["actual_label"] == label].head(100)  
    filtered_dfs.append(filtered_df)

df = pd.concat(filtered_dfs, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define models to be used for classification
models = [
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-3.5-turbo"
]

# Prepare to store evaluation results
results = []

def classify_text(input_text, prompt, model_name, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            match = re.search(r'\b[0-3]\b', response.choices[0].message.content)
            if match:
                return int(match.group(0))
            else:
                raise ValueError("No valid integer (0-3) found in response")
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

for model in models:
    for i, prompt in enumerate(classification_prompts):
        col_name = f"classification_{model}_prompt{i+1}"
        conf_matrix_file = f"confusion_matrix_{model}_prompt{i+1}.png"
        incorrect_csv_file = f"incorrect_classifications_{model}_prompt{i+1}.csv"

        if os.path.exists(conf_matrix_file) and os.path.exists(incorrect_csv_file):
            print(f"Skipping {model} - Prompt {i+1} as results already exist.")
            continue

        try:
            df[col_name] = df["text"].apply(lambda x: classify_text(x, prompt, model))
            predicted_labels = df[col_name].dropna().astype(int)
            actual_labels = df["actual_label"].dropna().astype(int)

            min_length = min(len(predicted_labels), len(actual_labels))
            predicted_labels = predicted_labels[:min_length]
            actual_labels = actual_labels[:min_length]

            conf_matrix = confusion_matrix(actual_labels, predicted_labels)
            accuracy = accuracy_score(actual_labels, predicted_labels)
            precision = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
            recall = recall_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
            f1 = f1_score(actual_labels, predicted_labels, average='weighted', zero_division=0)

            results.append({
                "Model": model,
                "Prompt": f"Prompt {i+1}",
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1, 4),
            })

            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(actual_labels), yticklabels=np.unique(actual_labels))
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model} - Prompt {i+1}\nAccuracy: {accuracy:.4f}")
            plt.savefig(conf_matrix_file)
            plt.close()

            incorrect_classifications = df[df[col_name] != df["actual_label"]]
            incorrect_classifications[["text", "actual_label", col_name]].to_csv(incorrect_csv_file, index=False)

            print(f"Metrics for {model} - Prompt {i+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        except Exception as e:
            print(f"Skipping classification for model {model} with prompt {i+1} due to error: {e}")

results_df = pd.DataFrame(results)
results_df.to_csv("classification_metrics.csv", index=False)
print("Final Performance Metrics:")
print(results_df)
df.to_csv("classified_results.csv", index=False)
print("Classification complete. Results, performance plots, and incorrect classifications saved.")
#%%

