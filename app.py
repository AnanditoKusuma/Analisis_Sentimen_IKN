import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, string, pickle, base64
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models import FastText
from wordcloud import WordCloud
import nltk

nltk.download('stopwords')

# Konfigurasi
st.set_page_config(page_title="Analisis Sentimen IKN", layout="wide", page_icon="📊")

# Mode tema
mode = st.sidebar.radio("🌓 Tema", ["🌞 Terang", "🌙 Gelap"], horizontal=True)
darkmode = (mode == "🌙 Gelap")

if darkmode:
    st.markdown("""
        <style>
        body, .stApp { background-color: #0e1117; color: white; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp { background-color: #ffffff; color: black; }
        </style>
    """, unsafe_allow_html=True)

# HEADER
col1, col2 = st.columns([1, 9])
with col1:
    st.image("LOGO UNISNU.png", width=90)
with col2:
    st.markdown("""
        <h1 style='margin-bottom:0;'>📊 Analisis Sentimen IKN</h1>
        <h5 style='margin-top:0; color:#00cc88;'>Metode: FastText + LightGBM | Akurasi: 83.27%</h5>
    """, unsafe_allow_html=True)

st.markdown("---")

# Load model
@st.cache_resource
def load_models():
    fasttext_model = FastText.load("fasttext_model.bin")
    with open("lightgbm_model.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    return fasttext_model, lgb_model

fasttext_model, lgb_model = load_models()

# Preprocessing
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

# Input manual
st.subheader("📝 Prediksi Manual Tweet")
input_text = st.text_area("Masukkan tweet:")

if st.button("🔍 Prediksi"):
    clean = clean_text(input_text)
    vec = vectorize(clean)
    hasil = lgb_model.predict([vec])[0]
    label = "Positif 😊" if hasil == 1 else "Negatif 😠"
    st.success(f"Prediksi Sentimen: **{label}**")

# Upload CSV
st.subheader("📂 Upload File CSV")
uploaded_file = st.file_uploader("Unggah file CSV dengan kolom 'tweet'", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'tweet' not in df.columns:
        st.error("❌ Kolom 'tweet' tidak ditemukan.")
    else:
        df['clean_tweet'] = df['tweet'].apply(clean_text)
        df['vector'] = df['clean_tweet'].apply(vectorize)
        X = np.vstack(df['vector'].values)
        df['prediksi'] = lgb_model.predict(X)
        df['sentimen'] = df['prediksi'].map({0: 'Negatif', 1: 'Positif'})

        # Ringkasan
        total = len(df)
        positif = sum(df['sentimen'] == 'Positif')
        negatif = sum(df['sentimen'] == 'Negatif')
        st.success(f"""
        ✅ Total Tweet: {total}  
        😊 Positif: {positif} ({(positif/total)*100:.1f}%)  
        😠 Negatif: {negatif} ({(negatif/total)*100:.1f}%)
        """)

        # Tabel hasil
        st.subheader("📋 Tabel Hasil Prediksi")
        st.dataframe(df[['tweet', 'sentimen']])

        # Visualisasi distribusi
        st.subheader("📊 Distribusi Sentimen")
        chart_type = st.radio("Pilih Grafik", ["Pie", "Bar"], horizontal=True)
        colors = ['#33cc99', '#ff4b4b']

        if chart_type == "Pie":
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            df['sentimen'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90,
                                                   colors=colors, textprops={'fontsize': 12},
                                                   wedgeprops={'edgecolor': 'white'}, ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.countplot(x='sentimen', data=df, palette=colors, ax=ax)
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)

        # Word Cloud
        st.subheader("☁️ Word Cloud")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 😊 Positif")
            positif = " ".join(df[df['sentimen'] == 'Positif']['clean_tweet'])
            wc_pos = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(positif)
            fig1, ax1 = plt.subplots()
            ax1.imshow(wc_pos, interpolation='bilinear')
            ax1.axis("off")
            st.pyplot(fig1)

        with col2:
            st.markdown("### 😠 Negatif")
            negatif = " ".join(df[df['sentimen'] == 'Negatif']['clean_tweet'])
            wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(negatif)
            fig2, ax2 = plt.subplots()
            ax2.imshow(wc_neg, interpolation='bilinear')
            ax2.axis("off")
            st.pyplot(fig2)

        # Download hasil
        st.subheader("⬇️ Unduh Hasil Prediksi")
        def get_csv_download_link(df):
            csv = df[['tweet', 'sentimen']].to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi.csv">📥 Klik untuk unduh CSV</a>'
            return href
        st.markdown(get_csv_download_link(df), unsafe_allow_html=True)
