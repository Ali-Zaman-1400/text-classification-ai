import pickle
import sys

# Load model
with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Get input text from command line
if len(sys.argv) < 2:
    print(" Please provide a text input")
    sys.exit(1)

text = sys.argv[1]

# Predict
X = vectorizer.transform([text])
prediction = model.predict(X)[0]

print(f" Predicted label: {prediction}")
