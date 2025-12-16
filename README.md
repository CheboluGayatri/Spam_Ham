# 📧 Spam and Ham Email Classifier using Machine Learning
# 📌 Project Overview

This project classifies messages as Spam or Ham (Not Spam) using a Machine Learning model. It uses a Naive Bayes classifier combined with TF-IDF vectorization to analyze text messages and predict whether a message is spam.

The project demonstrates a complete NLP based ML workflow, including text preprocessing, feature extraction, model training, evaluation, and deployment using a Streamlit web application.

# 🎯 Problem Statement

Spam messages are a major problem in emails and messaging platforms, causing security risks, scams, and poor user experience. Manual filtering is inefficient and inconsistent. This project aims to build a data driven solution that automatically classifies messages as spam or ham based on historical labeled data.

# 🧠 Machine Learning Approach

Model Used: Multinomial Naive Bayes

Reason: Efficient, fast, and highly effective for text classification problems

Feature Extraction: TF-IDF Vectorization

Task Type: Supervised Classification

# 📂 Dataset Information

Dataset Name: spam.csv

Target Variable: label (spam or ham)

Input Feature: Message text

Data Preprocessing

Converted text to lowercase

Removed punctuation and special characters

Removed missing values

Applied TF-IDF vectorization

Ensured consistent vectorizer usage during training and prediction

# ⚙️ Project Structure
Spam_Ham/
├── spam.csv                  # Dataset
├── app.py                    # Streamlit web application
├── ml_model.py               # Model training and evaluation
├── make_vectorizer.py        # TF-IDF vectorizer creation
├── spam.py                   # Prediction helper script
├── models/
│   ├── spam_model.pkl        # Trained Naive Bayes model
│   └── vectorizer.pkl        # Saved TF-IDF vectorizer
├── bg.jpg                    # Background image for UI
├── requirements.txt
└── README.md
🚀 Model Training

The dataset is split into training and testing sets using an 80:20 ratio.

Evaluation Metrics

Accuracy

After training:

The trained Naive Bayes model is saved as spam_model.pkl

The TF-IDF vectorizer is saved as vectorizer.pkl

These saved files are reused during prediction to ensure consistency.

# 🖥️ Web Application (Streamlit)

The Streamlit application allows users to:

Enter a custom message

Click a button to classify the message

Instantly view whether the message is Spam or Ham

UI Features

Custom background image

Glassmorphism styled card layout

Color coded prediction results

User friendly input validation

# 🛠️ Installation and Setup
1️⃣ Clone the Repository
git clone <https://github.com/CheboluGayatri/Spam_Ham.git>
cd Spam_Ham
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Train the Model
python ml_model.py

This will train the model and print evaluation metrics.

4️⃣ Create Vectorizer (if needed)
python make_vectorizer.py
5️⃣ Run the Streamlit App
streamlit run app.py
📈 Sample Output

SPAM → Displayed in red box

HAM → Displayed in green box

Predictions are based on learned word patterns from the training data.

# 🔐 Error Handling

Ensures model and vectorizer files exist before prediction

Prevents empty message submissions

Handles incorrect or malformed inputs gracefully

# 🔮 Future Improvements

Add precision, recall, and F1 score to evaluation

Use advanced models like Logistic Regression or SVM

Implement deep learning models (LSTM, Transformers)

Deploy using Streamlit Cloud or Docker

Convert the project into a REST API using FastAPI

# 📚 Skills Demonstrated

Natural Language Processing fundamentals

Text preprocessing and feature engineering

Supervised machine learning classification

Model evaluation and persistence

Streamlit based deployment

End to end ML project implementation
#  🌐 Live Demo
The application is deployed using Streamlit Cloud and is accessible at:
https://spamham-feaj3duta6rqjfprakg3da.streamlit.app/

Any updates pushed to the main branch are automatically reflected in the live app.
