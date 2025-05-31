import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

def training_nn(X_train, X_test, X_val, y_train, y_test, y_val):
    # Convert to arrays
    X_train = np.array(X_train.tolist())
    X_test = np.array(X_test.tolist())
    X_val = np.array(X_val.tolist())

    # One-hot encode labels
    num_classes = len(set(y_train))
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    vocab_size = np.max(X_train) + 1  # max token index + 1
    input_length = X_train.shape[1]

    # Build model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=input_length),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train
    model.fit(X_train, y_train_cat,
              validation_data=(X_val, y_val_cat),
              epochs=10,
              batch_size=32)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_cat)
    print(f"\nTest Accuracy: {accuracy:.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("C:/Users/tunde/PycharmProjects/NLP_Project/text_app_project/models/classiffier/emotions_classifier.h5")
    print("Model saved to models/text_classifier.h5")

    # Predict and compute confusion matrix
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
