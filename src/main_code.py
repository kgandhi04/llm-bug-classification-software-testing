#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import optuna
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from huggingface_hub import login
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset
print("Loading training and testing datasets...")
train_dataset_path = r'C:\Users\panka\OneDrive\Documents\GitHub\llm-bug-classification-software-testing\datafiles\train.csv'
test_dataset_path = r'C:\Users\panka\OneDrive\Documents\GitHub\llm-bug-classification-software-testing\datafiles\test.csv'
train_df = pd.read_csv(train_dataset_path)
test_df = pd.read_csv(test_dataset_path)
print("Datasets loaded successfully.")

# Hugging Face login
print("Logging into Hugging Face...")
hf_token = "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" #HF Token
login(token=hf_token)
print("Login successful.")

# Combine 'title' and 'body' into a single text field
train_df['text'] = train_df['title'] + " " + train_df['body']

# Encode labels
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['labels'])

# Sample 50,000 instances per label
train_df = train_df.groupby('label_encoded', group_keys=False).apply(lambda x: x.sample(min(len(x), 50000), random_state=42))

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['label_encoded'], test_size=0.2, random_state=42
)

# Tokenization function
def tokenize_data(texts, tokenizer, max_length=512):
    texts = [str(t) for t in texts]  # Ensure all inputs are strings
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenize_data(texts, tokenizer)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Define models
models = {
    "ModernBERT": "answerdotai/ModernBERT-base",
    "RoBERTa": "roberta-base"
}

# Train and evaluate each model
results = {}
for model_name, model_path in models.items():
    print(f"\nTraining {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(label_encoder.classes_))
    
    training_args = TrainingArguments(
        output_dir=f'./results_{model_name}',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f'./logs_{model_name}',
        report_to='none',
        save_strategy='epoch',
        save_total_limit=2,
        logging_first_step=True,
        logging_steps=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    predictions = trainer.predict(val_dataset)
    y_preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()
    
    # Save results
    report = classification_report(val_labels, y_preds, target_names=label_encoder.classes_, output_dict=True)
    results[model_name] = report
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(val_labels, y_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'conf_matrix_{model_name}.png')
    plt.close()

# FastText Model
print("\nTraining FastText Model...")
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_val_tfidf = vectorizer.transform(val_texts)

fasttext_model = LogisticRegression(max_iter=500)
fasttext_model.fit(X_train_tfidf, train_labels)
y_preds_fasttext = fasttext_model.predict(X_val_tfidf)

# Save results for FastText
report_fasttext = classification_report(val_labels, y_preds_fasttext, target_names=label_encoder.classes_, output_dict=True)
results["FastText"] = report_fasttext

# Confusion Matrix for FastText
conf_matrix_fasttext = confusion_matrix(val_labels, y_preds_fasttext)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_fasttext, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - FastText')
plt.savefig('conf_matrix_FastText.png')
plt.close()

# Save results
joblib.dump(results, "model_evaluation_results.pkl")

print("All models trained and evaluated successfully.")

#%%