import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Select only message column
texts = df['text']

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit vectorizer
vectorizer.fit(texts)

# Save vectorizer
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("vectorizer.pkl created successfully!")
