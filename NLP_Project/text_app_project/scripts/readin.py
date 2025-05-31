import os
import pandas as pd

def read_raw_data(file_name):
    # Define base path to raw data folder
    base_path = 'C:/Users/tunde/PycharmProjects/NLP_Project/text_app_project/data/raw/'

    # Join base path and file name to form full path
    file_path = os.path.join(base_path, file_name)

    # Read the CSV
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        print("CSV file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # print(data.head())
    label_counts = pd.crosstab(index=data['label'], columns='count')

    print(label_counts)
    
    return data