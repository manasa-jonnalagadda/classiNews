from flask import Flask, render_template, request
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Load saved vectorizer and model
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("news_classifier_nb.pkl")


def preprocess(text: str) -> str:
    return str(text).lower()


# -------- HOME PAGE (/) --------
TRENDING_ARTICLES = [
    {
        "title": "Government announces new education reform policy",
        "snippet": "The central government unveiled a comprehensive policy aimed at upgrading public schools and digital learning infrastructure.",
        "category": "politics",
    },
    {
        "title": "National team wins thrilling final over cricket match",
        "snippet": "Fans celebrated as the team chased down the target with a six on the last ball in a dramatic championship clash.",
        "category": "sports",
    },
    {
        "title": "Tech giant launches AI-powered smartphone lineup",
        "snippet": "The company introduced a new series of phones featuring on-device generative AI and enhanced camera processing.",
        "category": "technology",
    },
    {
        "title": "Blockbuster film breaks opening weekend records",
        "snippet": "The much-awaited movie drew massive crowds and set a new record at the global box office.",
        "category": "entertainment",
    },
    {
        "title": "Markets rally after strong quarterly earnings",
        "snippet": "Investors reacted positively as several blue-chip companies reported better-than-expected revenue and profit.",
        "category": "business",
    },
]


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", articles=TRENDING_ARTICLES)


# -------- CLASSIFIER PAGE (/classify) --------
@app.route("/classify", methods=["GET", "POST"])
def classify():
    prediction = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("news_text", "").strip()
        if user_text:
            clean = preprocess(user_text)
            vec = tfidf.transform([clean])
            prediction = model.predict(vec)[0]

    return render_template("classify.html", prediction=prediction, user_text=user_text)


if __name__ == "__main__":
    app.run(debug=True)
