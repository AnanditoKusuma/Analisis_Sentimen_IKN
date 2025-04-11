import pandas as pd
import numpy as np
import re
import string
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# === 1. Load dan bersihkan data ===
df = pd.read_csv("ikn.csv")

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\@\w+|\#\w+", '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = tokenizer.tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['clean_tweet'] = df['tweet'].apply(clean_text)
tokenized = [tweet.split() for tweet in df['clean_tweet']]

# === 2. Latih FastText ===
fasttext_model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, epochs=10)
fasttext_model.save("fasttext_model.bin")

# === 3. Vektorisasi tweet ===
def vectorize(tokens, model, vector_size=100):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)

X = np.array([vectorize(tweet, fasttext_model, 100) for tweet in tokenized])
y = df['sentiment'].map({'negative': 0, 'positive': 1})

# === 4. Train LightGBM ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Simpan model
with open("lightgbm_model.pkl", "wb") as f:
    pickle.dump(lgb_model, f)

# Akurasi
y_pred = lgb_model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("✅ Model disimpan sebagai: fasttext_model.bin & lightgbm_model.pkl")
