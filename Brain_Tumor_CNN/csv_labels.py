import os
import pandas as pd

yes_path = 'PATH'
no_path = 'PATH'
CSV_SAVE_PATH = ''

def get_file_labels(path, label):
    file_names = os.listdir(path)
    if not file_names:
        print(f"No files found in {path}")
        return []
    return [(file, label) for file in file_names if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

yes_files = get_file_labels(yes_path, 1)
no_files = get_file_labels(no_path, 0)

all_files = yes_files + no_files
df = pd.DataFrame(all_files, columns=['FileName', 'Target'])

df.to_csv(CSV_SAVE_PATH, index=False)