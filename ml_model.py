import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_dataset():
    df = pd.read_csv(
        "spam.csv",
        encoding="latin-1",
        sep=",",
        engine="python",
        on_bad_lines="skip"
    )

    df = df.iloc[:, :2]
    df.columns = ["label", "message"]
    df = df.dropna()

    return df


def clean_text(series):
    series = series.str.lower()
    series = series.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    return series


def train_model(df):
    df["message"] = clean_text(df["message"])

    X = df["message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # less restrictive vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 1)  # keep it simple
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # stronger smoothing helps with imbalance
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
    
    }

    return model, vectorizer, metrics


if __name__ == "__main__":
    df = load_dataset()
    model, vectorizer, metrics = train_model(df)
    print(metrics)
