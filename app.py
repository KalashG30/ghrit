from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Define the function used in the vectorizer
def word_split(inputs):
    character = []
    for i in inputs:
        character.append(i)
    return character

# Load model and vectorizer
model = joblib.load("password_strength_model_1.pkl")
vectorizer = joblib.load("tfidf_vectorizer_1.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        password = request.form["password"]
        transformed = vectorizer.transform([password])
        result = model.predict(transformed)[0]
        label = {0: "Weak", 1: "Medium", 2: "Strong"}.get(result, "Unknown")
        prediction = f"Password strength: {label}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
