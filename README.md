# Spam/Ham Email Classifier

A machine learning project to classify messages as **spam** or **ham** (not spam) using Python and scikit-learn.

---

## Project Overview

This project uses a **Naive Bayes classifier** with **TF-IDF vectorization** to detect spam messages. It includes scripts to train the model, save it, and test new messages.

---

# Project Structure

Spam_ham/
├─ app.py # Main script for predicting messages
├─ ml_model.py # Script to train the model
├─ make_vectorizer.py # Script to create the TF-IDF vectorizer
├─ spam.py # Test script for predictions
├─ spam.csv # Dataset of spam/ham messages
├─ models/
│ ├─ spam_model.pkl # Trained Naive Bayes model
│ └─ vectorizer.pkl # TF-IDF vectorizer
├─ bg.jpg # Optional image for UI
└─ pycache/ # Python cache files

yaml
Copy code

---

#  Features

- Text preprocessing (lowercasing, removing special characters)  
- TF-IDF vectorization of messages  
- Naive Bayes classification  
- Model and vectorizer persistence  
- Metrics: accuracy, precision, recall, F1 score  

---

# Requirements

- Python 3.10+  
- Packages:

```bash
pip install pandas scikit-learn
Usage
Clone the repository

bash
Copy code
git clone https://github.com/CheboluGayatri/Spam_Ham.git
cd Spam_Ham
Train the model

bash
Copy code
python ml_model.py
Trains the model on spam.csv and saves spam_model.pkl and vectorizer.pkl in models/.

Predict new messages

bash
Copy code
python spam.py
This will load the saved model and predict whether new messages are spam or ham.

GitHub Setup
Initialize and commit:

bash
Copy code
git init
git add .
git commit -m "Initial commit"
Add remote:

bash
Copy code
git remote add origin https://github.com/CheboluGayatri/Spam_Ham.git
Push to GitHub main branch:

bash
Copy code
git branch -M main
git push -u origin main --force
Notes
Dataset: spam.csv must be in the project root

Windows may show LF → CRLF warnings; this is normal

Model is trained on a small dataset; some misclassifications may occur

License
This project is open source and free to use.
