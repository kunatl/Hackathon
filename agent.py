import joblib
from rag_engine import suggest_resolution

# Load classifier
model = joblib.load("ticket_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

CONF_THRESHOLD = 0.6

routing_map = {
    "Network": "Network Team",
    "Database": "DB Team",
    "Security": "Security Team",
    "Application": "App Team",
    "Infrastructure": "Infra Team",
    "Access Management": "IAM Team"
}


def classify_ticket(text):
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    
    prediction = model.classes_[probs.argmax()]
    confidence = probs.max()
    
    return prediction, confidence


def agent_decision(ticket_text):
    # Step 1: Classification
    category, confidence = classify_ticket(ticket_text)
    
    # Step 2: RAG (Gemini-powered)
    resolution, similar_tickets = suggest_resolution(ticket_text)
    
    # Step 3: Routing
    department = routing_map.get(category, "Support Team")
    
    # Step 4: Decision Logic
    repeat_count = sum(
        1 for t in similar_tickets if t["category"] == category
    )
    
    if confidence < CONF_THRESHOLD:
        action = "ESCALATE"
    elif repeat_count >= 2:
        action = "AUTOMATE_THIS_ISSUE"
    else:
        action = "AUTO_ROUTE_AND_RESOLVE"
    
    return {
        "ticket": ticket_text,
        "category": category,
        "confidence": round(confidence, 2),
        "assigned_to": department,
        "suggested_resolution": resolution,
        "action": action
    }


if __name__ == "__main__":
    ticket = input("Enter ticket:\n")
    
    result = agent_decision(ticket)
    
    print("\n🤖 AI Agent Output:\n")
    for k, v in result.items():
        print(f"{k.upper()}: {v}")