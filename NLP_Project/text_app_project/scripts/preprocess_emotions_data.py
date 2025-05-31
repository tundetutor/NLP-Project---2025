from readin import read_raw_data

import re
import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def preprocess():
    # Step 1: Load data
    data = read_raw_data('emotions_merged.csv')

    # Step 2: Clean text
    data['text_clean'] = data['text'].apply(clean_text)

    # Step 3: Tokenization
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(data['text_clean'])
    sequences = tokenizer.texts_to_sequences(data['text_clean'])

    # Step 4: Padding
    max_len = max(len(seq) for seq in sequences)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # Step 5: Save the tokenizer
    with open("C:/Users/tunde/PycharmProjects/NLP_Project/text_app_project/models/classiffier/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved to: text_app_project/models/classiffier/tokenizer.pkl")

    labels = {
        'sadness': 0,
        'love': 1,
        'anger': 2,
        'joy': 3,
        'fear': 4,
        'surprise': 5
    }

    # Step 6: Create final DataFrame
    final_df = pd.DataFrame({
        'text': padded.tolist(),
        'label': data['label'].map(labels)
    })

    print(final_df.head())
    print("Unique labels:")
    print(final_df['label'].unique())

    print("Label distribution:")
    print(final_df['label'].value_counts())

    return final_df
