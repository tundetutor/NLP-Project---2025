import os
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm
from preprocess_corpus import create_corpus

def encode_sequence(seq_list, char_to_int, unk_token=0):
    return np.array([[char_to_int.get(char, unk_token) for char in seq] for seq in seq_list])

def encode_labels(label_seq, char_to_int, unk_token=0):
    return np.array([char_to_int.get(char, unk_token) for char in label_seq])

def training_gru_network(data_dir):
    print("Loading and preprocessing data...")
    x_train, y_train, x_val, y_val, x_test, y_test = create_corpus(data_dir, window_size=50)

    print("Creating character vocabulary...")
    vocab = sorted(set(np.concatenate(x_train).tolist() + y_train.tolist()))
    char_to_int = {char: idx for idx, char in enumerate(vocab)}
    int_to_char = {idx: char for char, idx in char_to_int.items()}
    vocab_size = len(vocab)

    print("Encoding character sequences...")
    x_train_enc = encode_sequence(x_train, char_to_int)
    x_val_enc = encode_sequence(x_val, char_to_int)
    x_test_enc = encode_sequence(x_test, char_to_int)
    y_train_enc = encode_labels(y_train, char_to_int)
    y_val_enc = encode_labels(y_val, char_to_int)
    y_test_enc = encode_labels(y_test, char_to_int)

    embedding_dim = 128
    rnn_units = 100

    print("Building GRU model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        tf.keras.layers.GRU(rnn_units, return_sequences=True),
        tf.keras.layers.GRU(rnn_units),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    epochs = 3
    batch_size = 64

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.fit(
            x_train_enc, y_train_enc,
            batch_size=batch_size,
            epochs=1,
            validation_data=(x_val_enc, y_val_enc)
        )

        # === Text Generation ===
        start_idx = np.random.randint(0, len(x_test_enc))
        seed_text = "".join([int_to_char[i] for i in x_test_enc[start_idx]])
        generated = seed_text

        for _ in tqdm(range(100), desc="Generating text"):
            input_seq = [char_to_int.get(char, 0) for char in generated[-50:]]
            input_seq = np.expand_dims(input_seq, axis=0)
            probs = model.predict(input_seq, verbose=0)[0]
            next_index = np.random.choice(len(probs), p=probs)
            generated += int_to_char[next_index]

        print(f"\nSample Generated Text:\n{generated}")

    model_dir = "text_app_project/models/text_gen"
    os.makedirs(model_dir, exist_ok=True)

    model.save(os.path.join(model_dir, "text_gru_model.h5"))

    with open(os.path.join(model_dir, "char_mapping.pkl"), "wb") as f:
        pickle.dump({'char_to_int': char_to_int, 'int_to_char': int_to_char}, f)

    print("Model and character mappings saved.")
