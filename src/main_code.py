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

# Load the dataset
print("Loading training and testing datasets...")
train_dataset_path = r'C:\Users\panka\OneDrive\Documents\GitHub\llm-bug-classification-software-testing\datafiles\train.csv'
test_dataset_path = r'C:\Users\panka\OneDrive\Documents\GitHub\llm-bug-classification-software-testing\datafiles\test.csv'
train_df = pd.read_csv(train_dataset_path)
test_df = pd.read_csv(test_dataset_path)
print("Datasets loaded successfully.")

# Hugging Face login
print("Logging into Hugging Face...")
hf_token = "<HF TOKEN>"
login(token=hf_token)
print("Login successful.")

# Preprocessing
print("Combining 'title' and 'body' into a single text field...")
train_df['text'] = train_df['title'] + " " + train_df['body']
print("Text combination completed.")

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['labels'])
print("Labels encoded successfully.")

# Split data
print("Splitting data into training and validation sets...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['label_encoded'], test_size=0.2, random_state=42
)
print(f"Data split completed: {len(train_texts)} training samples and {len(val_texts)} validation samples.")

# Tokenization
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
print("Tokenizer loaded successfully.")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=8192):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

print("Creating dataset objects...")
train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)
print("Datasets created successfully.")

# Load model
print("Loading pre-trained model...")
model = AutoModelForSequenceClassification.from_pretrained('answerdotai/ModernBERT-base', num_labels=len(label_encoder.classes_))
print("Model loaded successfully.")

# Training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to='none'  # Disable wandb
)
print("Training arguments set.")

# Trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
print("Trainer initialized successfully.")

# Train the model
print("Starting training process...")
trainer.train()
print("Training completed successfully.")

# Evaluate the model
print("Evaluating the model on validation dataset...")
predictions = trainer.predict(val_dataset)
y_preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()
print("Model evaluation completed.")

# Confusion Matrix
print("Generating confusion matrix...")
conf_matrix = confusion_matrix(val_labels, y_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
print("Confusion matrix displayed.")
#%%
# Classification report
print("Generating classification report...")
print(classification_report(val_labels, y_preds, target_names=label_encoder.classes_))
print("Classification report displayed.")

# Plot training loss and accuracy
print("Extracting training history...")
history = trainer.state.log_history
loss_values = [x['loss'] for x in history if 'loss' in x]
eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

print("Plotting training and validation loss...")
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, label='Training Loss')
plt.plot(epochs, eval_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
print("Loss plots displayed.")

print("Script execution completed.")
#%%