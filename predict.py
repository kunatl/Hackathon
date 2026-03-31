import joblib

# Load model
model = joblib.load("ticket_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_ticket(text):
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    prediction = model.classes_[probs.argmax()]
    confidence = probs.max()
    return prediction, confidence

# 🔹 Take input from user
ticket = input("Enter your ticket description:\n")

category, confidence = predict_ticket(ticket)

print(f"\nPredicted Category: {category}")
print(f"Confidence: {confidence:.2f}")