import os
import numpy as np
from tqdm import tqdm
import psutil

def print_memory_usage(note=""):
    mem = psutil.Process().memory_info().rss / 1024**2
    print(f"[{note}] Memory usage: {mem:.2f} MB")

def merge_txt(dir):
    """
    Reads and merges all .txt files into a single lowercase corpus list.
    """
    corpus_list = []
    individual_lengths = {}

    print("\nReading book files...")
    for filename in tqdm(os.listdir(dir), desc="Reading .txt files"):
        if filename.endswith(".txt"):
            file_path = os.path.join(dir, filename)
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
                individual_lengths[filename] = len(text)
                corpus_list.append(text)

    print("\nIndividual File Lengths:")
    for fname, length in individual_lengths.items():
        print(f"{fname}: {length} characters")

    # Merge and lowercase
    corpus = ' '.join(corpus_list).lower()
    merged_length = len(corpus)
    sum_of_individuals = sum(individual_lengths.values())

    print("\nCorpus Check:")
    print(f"Sum of individual file lengths: {sum_of_individuals}")
    print(f"Length of merged corpus:        {merged_length}")
    print("Character count matches!" if sum_of_individuals == merged_length else " Mismatch due to extra spaces.")

    return list(corpus)

def list_to_array_with_progress(lst, name="Array", chunk_size=100_000):
    """
    Converts a list into a NumPy array with a tqdm progress bar.
    """
    length = len(lst)
    is_sequence = isinstance(lst[0], (list, tuple, np.ndarray))

    if is_sequence:
        shape = (length, len(lst[0]))
        dtype = type(lst[0][0])
    else:
        shape = (length,)
        dtype = type(lst[0])

    arr = np.empty(shape, dtype=dtype)

    for i in tqdm(range(0, length, chunk_size), desc=f"Converting {name} to np.array"):
        end = i + chunk_size
        arr[i:end] = lst[i:end]

    return arr

def create_sliding_window(corpus, window_size):
    """
    Create overlapping windows from the corpus.
    Each X is a sequence, and Y is the next character.
    """
    x = []
    y = []

    print("\n🪟 Creating sliding windows...")
    for i in tqdm(range(len(corpus) - window_size), desc="Building X/Y samples"):
        x.append(corpus[i:i + window_size])
        y.append(corpus[i + window_size])

    print_memory_usage("Before np.array conversion")

    x = list_to_array_with_progress(x, name="X")
    y = list_to_array_with_progress(y, name="Y")

    print_memory_usage("After np.array conversion")
    return x, y

def data_split(x, y, train_frac=0.8, val_frac=0.1):
    """
    Splits data into train, validation, and test sets.
    """
    total_len = len(x)
    train_end = int(train_frac * total_len)
    val_end = int((train_frac + val_frac) * total_len)

    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]

    print("\nData Split Shapes:")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_val shape:   {x_val.shape}")
    print(f"y_val shape:   {y_val.shape}")
    print(f"x_test shape:  {x_test.shape}")
    print(f"y_test shape:  {y_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test

def create_corpus(dir, window_size=50):
    """
    Full preprocessing pipeline: merge, tokenize, window, split.
    """
    corpus = merge_txt(dir)
    x, y = create_sliding_window(corpus, window_size)
    return data_split(x, y)
