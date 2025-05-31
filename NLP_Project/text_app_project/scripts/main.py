from preprocess_emotions_data import preprocess
from training_nn import training_nn
from data_split import data_split
from preprocess_corpus import create_corpus
from training_gru_n import training_gru_network

def train_classifier():
    data = preprocess()
    X_train, X_test, X_val, y_train, y_test, y_val = data_split(data, 0.8)

    training_nn(X_train, X_test, X_val, y_train, y_test, y_val)

def train_text_gen():
    training_gru_network("C:/Users/tunde/PycharmProjects/NLP_Project/text_app_project/data/raw/books")

def main():
    # train_text_gen()
    train_classifier()

if __name__ == "__main__":
    main()