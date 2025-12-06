import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Emotion Detective", page_icon="🕵️‍♀️", layout="wide")

# --- DOWNLOAD NLTK RESOURCES ---
# We do this inside a try-except block or check to prevent re-downloading
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- PREPROCESSING FUNCTION ---
def clean_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Remove numbers
    text = re.sub(r'\d+', '', text)
    # 4. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # 5. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # 6. Remove emojis/special chars (keeping ASCII only)
    text = ''.join([i for i in text if i.isascii()])
    
    # 7. Remove Stopwords
    stop_words = set(stopwords.words('english'))
    words_to_keep = {
        # Negations (Critical)
        'no', 'not', 'nor', 'neither', 'never', 'none', "don't", "aren't", "couldn't", "didn't", "doesn't", "hadn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
        # Intensity (Important)
        'very', 'too', 'so', 'more', 'most', 'just',
        # Contrast (Context)
        'but', 'however', 'against'
    }
    final_stop_words = stop_words - words_to_keep
    
    words = text.split()
    filtered_words = [w for w in words if w not in final_stop_words]
    
    return ' '.join(filtered_words)

# --- MODEL TRAINING AND CACHING ---
@st.cache_resource
def load_and_train_model():
    # Load Data
    try:
        df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])
    except FileNotFoundError:
        st.error("Error: 'train.txt' not found. Please place the dataset in the same directory.")
        return None, None

    # Clean Data
    df['clean_text'] = df['text'].apply(clean_text)

    # Label Mapping (Dynamic)
    unique_emotions = df['emotion'].unique()
    label_map = {emo: i for i, emo in enumerate(unique_emotions)}
    inv_label_map = {i: emo for emo, i in label_map.items()}
    
    df['label'] = df['emotion'].map(label_map)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], 
        df['label'], 
        test_size=0.20, 
        random_state=42
    )

    # Create Pipeline (Vectorization + Model)
    # Using Logistic Regression + CountVectorizer as it had the best accuracy (~88%) in  notebook
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    pipeline.named_steps['vectorizer'].transform(X_test)
    ypred = pipeline.predict(X_test)
    accu=accuracy_score(y_test,ypred)
    
    
    return pipeline, inv_label_map ,accu

# --- LOAD MODEL ---
with st.spinner("Training AI Model... (This only happens once)"):
    model, emotion_map,accu = load_and_train_model()

# --- EMOJI MAPPING FOR UI ---
emoji_dict = {
    "joy": "😂",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😱",
    "love": "❤️",
    "surprise": "😲"
}

# --- UI LAYOUT ---
st.title("🧠 AI Emotion Detector")
st.markdown("""
This app uses **Machine Learning (Logistic Regression)** to analyze text and predict the underlying emotion.
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Type your text below:")
    user_input = st.text_area("Example: I was so happy when I saw the surprise party!", height=150)
    
    predict_btn = st.button("Analyze Emotion", type="primary")

with col2:
    st.subheader("Model Info")
    st.info(f"Accuracy on Test Data: **~{accu:.2%}**")
    st.markdown("**Classes:**")
    st.write(", ".join([f"{k} {v}" for k, v in emoji_dict.items()]))

# --- PREDICTION LOGIC ---
if predict_btn and user_input and model:
    # 1. Clean Input
    cleaned_input = clean_text(user_input)
    
    if not cleaned_input:
        st.warning("Please enter meaningful text (removing stopwords left nothing to analyze).")
    else:
        # 2. Predict
        prediction_idx = model.predict([cleaned_input])[0]
        prediction_label = emotion_map[prediction_idx]
        
        # 3. Probabilities
        probs = model.predict_proba([cleaned_input])[0]
        prob_df = pd.DataFrame({
            "Emotion": [emotion_map[i] for i in range(len(probs))],
            "Confidence": probs
        }).sort_values(by="Confidence", ascending=False)

        # --- DISPLAY RESULTS ---
        st.divider()
        
        r1, r2 = st.columns([1, 2])
        
        with r1:
            st.success("Prediction Complete!")
            emoji = emoji_dict.get(prediction_label, "🤖")
            st.markdown(f"<h1 style='text-align: center; font-size: 80px;'>{emoji}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>{prediction_label.title()}</h2>", unsafe_allow_html=True)
        
        with r2:
            st.subheader("Confidence Scores")
            # Using Streamlit's native bar chart
            st.bar_chart(prob_df.set_index("Emotion"))
            
        with st.expander("See Text Preprocessing Details"):
            st.write("**Original:**", user_input)
            st.write("**Cleaned:**", cleaned_input)

# --- SIDEBAR INFO ---
st.sidebar.title("About")
st.sidebar.write("This project implements NLP techniques including:")
st.sidebar.markdown("- **Text Cleaning** (Regex, NLTK)")
st.sidebar.markdown("- **Vectorization** (CountVectorizer)")
st.sidebar.markdown("- **Classification** (Logistic Regression)")