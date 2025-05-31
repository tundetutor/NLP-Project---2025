from text_app_project.app import app

from flask import render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

# === Emotion Classifier Setup ===
label_map = {
    0: 'sadness',
    1: 'love',
    2: 'anger',
    3: 'joy',
    4: 'fear',
    5: 'surprise'
}

clf_model_path = os.path.join("text_app_project", "models", "classiffier", "emotions_classifier.h5")
clf_model = load_model(clf_model_path)

tokenizer_path = os.path.join("text_app_project", "models", "classiffier", "tokenizer.pkl")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50  # sequence length for padding

# === Text Generator Setup ===
gen_model_path = os.path.join("text_app_project", "models", "text_gen", "text_gru_model.h5")
gen_model = load_model(gen_model_path)

mapping_path = os.path.join("text_app_project", "models", "text_gen", "char_mapping.pkl")
with open(mapping_path, "rb") as f:
    mapping = pickle.load(f)

char_to_int = mapping['char_to_int']
int_to_char = mapping['int_to_char']


# === Routes ===

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        input_text = request.form["text_input"]
        if input_text.strip():
            seq = tokenizer.texts_to_sequences([input_text])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
            pred = np.argmax(clf_model.predict(padded), axis=1)[0]
            result = label_map[pred]
        else:
            result = "Please enter a sentence."
    return render_template("index.html", mode="classifier", result=result)


@app.route("/generator", methods=["GET", "POST"])
def generator():
    generated_text = ""
    if request.method == "POST":
        seed_text = request.form["text_input"]
        if seed_text.strip():
            generated_text = seed_text
            for _ in range(100):  # generate 100 characters
                input_seq = [char_to_int.get(char, 0) for char in generated_text[-50:]]
                input_seq = np.expand_dims(input_seq, axis=0)
                probs = gen_model.predict(input_seq, verbose=0)[0]
                next_index = np.random.choice(len(probs), p=probs)
                next_char = int_to_char[next_index]
                generated_text += next_char
        else:
            generated_text = "Please enter a starting phrase."
    return render_template("index.html", mode="generator", result=generated_text)
