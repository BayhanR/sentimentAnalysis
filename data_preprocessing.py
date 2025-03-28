import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from data_balance import balanced_df

# balanced_df=pd.read("balanced_sentiment_dataset.csv") satırını değiştirelim
balanced_df = pd.read_csv("balanced_sentiment_dataset.csv")


def clean_text(text):
    text = text.lower()  # Küçük harfe çevir
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları kaldır
    return text


# Yorumları temizle
balanced_df['label'] = balanced_df['label'].apply(clean_text)

# 2. Sayısallaştırma (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(balanced_df['label']).toarray()
y = balanced_df['label'].values

# 3. Eğitim ve Test Seti Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve test boyutları
print("Eğitim seti boyutu:", X_train.shape)
print("Test seti boyutu:", X_test.shape)

# Eğitim ve test setlerini pandas DataFrame'e çevir
train_df = pd.DataFrame(X_train)
train_df['label'] = y_train
test_df = pd.DataFrame(X_test)
test_df['label'] = y_test

# CSV dosyalarına kaydet
train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)