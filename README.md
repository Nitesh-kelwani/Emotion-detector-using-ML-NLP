🧠 AI Emotion Detector
  A real-time web application that analyzes text and predicts the underlying emotion using Machine Learning and Natural Language Processing (NLP). Built with Python and Streamlit.

📌 Overview
  This application takes user text input, processes it using NLP techniques (handling negations and context), and classifies it into one of six emotions:
  
  Joy 😂
  Sadness 😢
  Anger 😡
  Fear 😨
  Love ❤️
  Surprise 😲
  
  It provides visual feedback on the prediction confidence using interactive charts.

Features
  
  Smart Preprocessing: Custom text cleaning that removes noise but preserves critical context words like negations (not, no, never) and intensifiers (very, so).
  N-Gram Analysis: Uses Bigrams to understand phrases like "not happy" vs "happy".
  Real-Time Prediction: Instant analysis using a pre-trained Logistic Regression model.
  Confidence Scores: Visual bar charts showing the probability of each emotion.
  Explainability: An "Under the Hood" view showing exactly how the AI cleaned and processed the text.

Tech Stack
  Python (3.8+)
  Streamlit (Web UI)
  Scikit-Learn (Machine Learning Pipeline)
  NLTK (Natural Language Processing)
  Pandas & NumPy (Data Manipulation)
  Altair (Data Visualization)

Project Structure
  ├── app.py              # Main application script (UI + Logic)
  ├── train.txt           # Training dataset (Text ; Emotion)
  ├── requirements.txt    # Python dependencies
  └── README.md           # Project documentation


Installation & Setup
  1.Clone the repository (or download the files):
      https://github.com/Nitesh-kelwani/Emotion-detector-using-ML-NLP.git
  2.Ensure you have the dataset: Make sure the train.txt file is located in the root directory. This file should be formatted with text and labels separated by a semicolon (;).
  3.Install Dependencies: It is recommended to use a virtual environment.
      pip install -r requirements.txt

How to Run
  Execute the following command in your terminal:
      streamlit run app.py
  The application will launch in your default web browser at http://localhost:8501.

Model & Preprocessing Details
  1. Text Cleaning
    The app uses a custom cleaning function that:
    Converts text to lowercase.
    Removes special characters, URLs, and HTML tags.
    Selectively removes stopwords: Unlike standard cleaners, this app keeps words like "not", "but", "very", and "against" to maintain sentiment context.
  2. Vectorization
    We use CountVectorizer with ngram_range=(1,2). This creates features for single words ("happy") and pairs of words ("not happy"), allowing the model to distinguish between positive and negative contexts.
  3. Classification
    The model uses Logistic Regression, which offers a great balance of speed and accuracy (~88% on the test set) for text classification tasks.

🤝 Contributing
  Contributions are welcome! Feel free to open an issue or submit a pull request if you have ideas for improvements.

📜 License
  This project is open-source. Feel free to use it for educational purposes.
