import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("high_quality_it_tickets.csv")

# Combine title + description
df["text"] = df["title"] + " " + df["description"]

# Features & labels
X = df["text"]
y = df["category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "ticket_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model and vectorizer saved!")