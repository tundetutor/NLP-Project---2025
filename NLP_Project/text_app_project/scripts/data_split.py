from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import numpy as np

def print_distribution(name, labels):
    counter = Counter(labels)
    total = sum(counter.values())
    print(f"\n{name} set distribution:")
    for label, count in counter.items():
        percent = 100 * count / total
        print(f"  Class {label}: {count} ({percent:.2f}%)")

def data_split(data, train_ratio=0.8):
    # Separate features and labels
    X = data['text']
    y = data['label']

    # First split: Train and Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_ratio, stratify=y, random_state=42
    )

    # Second split: Temp into Val and Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Optional: print class distributions
    # print_distribution("Train", y_train)
    # print_distribution("Validation", y_val)
    # print_distribution("Test", y_test)

    # Convert to NumPy arrays
    X_train = np.array(X_train.tolist())
    X_val = np.array(X_val.tolist())
    X_test = np.array(X_test.tolist())

    y_train = np.array(y_train.tolist())
    y_val = np.array(y_val.tolist())
    y_test = np.array(y_test.tolist())

    return X_train, X_test, X_val, y_train, y_test, y_val
