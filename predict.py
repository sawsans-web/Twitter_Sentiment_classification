import os
import pickle
import re
import emoji
import contractions
import argparse
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk_packages = [
    "stopwords",
    "wordnet",
    "omw-1.4",  
    "averaged_perceptron_tagger",
    "punkt"
]

for pkg in nltk_packages:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

try:
    import gdown
except ImportError:
    raise ImportError("You need to install gdown: pip install gdown")

FILE_ID = "1KE7kVAh8hoHxUuWOYXqUGfAayYA8SAyJ"
VECTORIZER_PATH = "vectorizer.pkl"
MODEL_PATH = "model.pkl"

def download_vectorizer():
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Downloading vectorizer from Google Drive...")
    gdown.download(url, VECTORIZER_PATH, quiet=False)
    print("Download complete.")

if not os.path.exists(VECTORIZER_PATH):
    download_vectorizer()

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please download it from Github or contact the repo owner.")

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

stop_words = set(stopwords.words('english'))
negations = {"not", "nor", "no", "n't", "weren't", "weren", "wasn't", "isn't", "aren't", "ain't"}
stop_words = stop_words - negations
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', lambda x: re.sub(r'([A-Z])', r' \1', x.group()[1:]).lower().strip(), text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = emoji.replace_emoji(text, replace=lambda e: emoji.demojize(e).replace(":", " ").replace("_", " "))
    words = []
    for word in text.split():
        if len(word) > 1 and word not in stop_words:
            lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
            words.append(lemma)
    return " ".join(words)

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    if vec.nnz == 0:  
        return "Input outside trained vocabulary"
    y_pred = model.predict(vec)[0]
    return "Positive" if y_pred == 1 else "Negative"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Predictor CLI")
    parser.add_argument("--input", type=str, required=True, help="Text or path to text file for prediction")
    args = parser.parse_args()

    inp = args.input

    if os.path.exists(inp) and os.path.isfile(inp):
        with open(inp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    result = predict_sentiment(line)
                    print(f"{line} -> {result}")
    else:
        result = predict_sentiment(inp)
        print("Predicted sentiment:", result)

