#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
print("Loading training dataset...")
train_dataset_path = r'C:\Users\panka\OneDrive\Documents\GitHub\llm-bug-classification-software-testing\datafiles\train.csv'
train_df = pd.read_csv(train_dataset_path)
print("Dataset loaded successfully.")

# Preprocessing
print("Combining 'title' and 'body' into a single text field...")
train_df['text'] = train_df['title'] + " " + train_df['body']
train_df['text'] = train_df['text'].fillna('')  # Handle NaN values
print("Text combination completed.")

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['labels'])
print("Labels encoded successfully.")

# Display class distribution before balancing
plt.figure(figsize=(8, 5))
sns.countplot(x=train_df['labels'], order=train_df['labels'].value_counts().index, palette='viridis')
plt.xticks(rotation=45)
plt.title("Class Distribution Before Balancing")
plt.show()

# Balance dataset (25,000 samples per category)
print("Balancing the dataset...")
min_samples = 25000  # Fixed sample size per category
balanced_df = train_df.groupby('labels', group_keys=False).apply(lambda x: x.sample(min(len(x), min_samples)))
balanced_df = balanced_df.reset_index(drop=True)
print(f"Balanced dataset size: {len(balanced_df)}")

# Display class distribution after balancing
plt.figure(figsize=(8, 5))
sns.countplot(x=balanced_df['labels'], order=balanced_df['labels'].value_counts().index, palette='coolwarm')
plt.xticks(rotation=45)
plt.title("Class Distribution After Balancing")
plt.show()

# Function to clean and get most common words per category
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

def get_most_common_words(text_series, n=20):
    all_words = []
    for text in text_series:
        all_words.extend(clean_text(text))
    return Counter(all_words).most_common(n)

# Word frequency analysis per category
word_freq_per_category = {}
for category in balanced_df['labels'].unique():
    word_freq_per_category[category] = get_most_common_words(balanced_df[balanced_df['labels'] == category]['text'])

# Display top 10 frequent words per category
for category, words in word_freq_per_category.items():
    print(f"\nMost Common Words in '{category}' Issues:")
    print(words[:10])

# Generate Word Clouds per Category
print("Generating word clouds...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, category in enumerate(balanced_df['labels'].unique()):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
        ' '.join([' '.join(clean_text(text)) for text in balanced_df[balanced_df['labels'] == category]['text']])
    )
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].set_title(f"Word Cloud - {category}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# Text length analysis
balanced_df['text_length'] = balanced_df['text'].apply(lambda x: len(clean_text(x)))
plt.figure(figsize=(10, 6))
sns.boxplot(x=balanced_df['labels'], y=balanced_df['text_length'], palette='Set2')
plt.xticks(rotation=45)
plt.title("Text Length Distribution per Category")
plt.show()

print("Exploratory Data Analysis (EDA) completed successfully.")

#%%