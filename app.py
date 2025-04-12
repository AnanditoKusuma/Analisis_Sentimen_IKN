import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pickle
from gensim.models import FastText
import nltk

# NLTK
nltk.download("stopwords")

# ---------------------- Config Awal ---------------------- #
st.set_page_config(
    page_title="Analisis Sentimen IKN",
    layout="wide",
    page_icon="📊"
)

# ---------------------- Sidebar Tema ---------------------- #
with st.sidebar:
    st.title("⚙️ Pengaturan")
    tema = st.radio("🎨 Pilih Tema", ["Terang", "Gelap"])
    visual = st.radio("📊 Pilih Visualisasi", ["Bar Chart", "Pie Chart"])

# ---------------------- Custom CSS ---------------------- #
dark_mode = tema == "Gelap"

st.markdown(
    f"""
    <style>
    body {{
        background-color: {"#0e1117" if dark_mode else "#ffffff"};
        color: {"#f0f2f6" if dark_mode else "#000000"};
    }}
    .stApp {{
        background-color: {"#0e1117" if dark_mode else "#ffffff"};
        color: {"#f0f2f6" if dark_mode else "#000000"};
    }}
    h1, h2, h3, h4, h5, h6, .stText {{
        color: {"#f0f2f6" if dark_mode else "#000000"};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- Header ---------------------- #
col_logo, col_judul = st.columns([0.1, 0.9])
with col_logo:
    st.image("LOGO UNISNU.png", width=100)
with col_judul:
    st.title("📊 Aplikasi Analisis Sentimen IKN dengan FastText + LightGBM")

# ---------------------- Load Model ---------------------- #
@st.cache_resource
def load_models():
    fasttext_model = FastText.load("fasttext_model.bin")
    with open("lightgbm_model.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    return fasttext_model, lgb_model

fasttext_model, lgb_model = load_models()

# ---------------------- Preprocessing ---------------------- #
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

def vectorize(text):
    tokens = text.split()
    vecs = [fasttext_model.wv[w] for w in tokens if w in fasttext_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(fasttext_model.vector_size)

# ---------------------- Input Manual ---------------------- #
st.subheader("✍️ Input Manual Tweet")
input_text = st.text_area("Masukkan tweet Anda:")

if st.button("Prediksi Sentimen"):
    clean = clean_text(input_text)
    vektor = vectorize(clean)
    hasil = lgb_model.predict([vektor])[0]
    label = "Positif 😊" if hasil == 1 else "Negatif 😠"
    st.success(f"Hasil Prediksi Sentimen: **{label}**")

# ---------------------- Upload CSV ---------------------- #
st.subheader("📂 Upload File CSV")
uploaded_file = st.file_uploader("Upload file CSV (harus ada kolom 'tweet')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    df['vector'] = df['clean_tweet'].apply(lambda x: vectorize(x))
    X = np.vstack(df['vector'].values)
    df['prediksi'] = lgb_model.predict(X)
    df['sentimen'] = df['prediksi'].map({0: 'Negatif', 1: 'Positif'})

    st.subheader("📄 Hasil Prediksi")
    st.dataframe(df[['tweet', 'sentimen']])

    # Visualisasi Distribusi Sentimen
    st.subheader("📊 Distribusi Sentimen")

    fig, ax = plt.subplots(figsize=(7, 4))
    if visual == "Bar Chart":
        sns.countplot(x='sentimen', data=df, palette='Set2', ax=ax)
        ax.set_title("Bar Chart Sentimen")
    else:
        df['sentimen'].value_counts().plot.pie(
            autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=90, ax=ax
        )
        ax.set_ylabel('')
        ax.set_title("Pie Chart Sentimen")
    st.pyplot(fig)

    # WordCloud
    st.subheader("☁️ Word Cloud")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Positif**")
        positif = " ".join(df[df['sentimen'] == 'Positif']['clean_tweet'])
        wc_pos = WordCloud(width=400, height=300, background_color='white' if not dark_mode else 'black').generate(positif)
        st.image(wc_pos.to_array(), use_container_width=True)

    with col2:
        st.markdown("**Negatif**")
        negatif = " ".join(df[df['sentimen'] == 'Negatif']['clean_tweet'])
        wc_neg = WordCloud(width=400, height=300, background_color='white' if not dark_mode else 'black').generate(negatif)
        st.image(wc_neg.to_array(), use_container_width=True)

    # Unduh Hasil
    st.subheader("⬇️ Unduh Hasil Prediksi")
    csv = df[['tweet', 'sentimen']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", data=csv, file_name='hasil_prediksi.csv', mime='text/csv')

# ---------------------- Akurasi Model ---------------------- #
st.sidebar.markdown("---")
st.sidebar.markdown("📈 **Akurasi Model**")
st.sidebar.success("🎯 Akurasi: 83.2% (FastText + LightGBM)")
