import os
import random
import pandas as pd
import sys

def select_random_files_to_csv(input_dir, output_path, n = 150):
    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            rel_dir = os.path.relpath(root, input_dir)
            print(rel_dir)
            rel_file = os.path.join(rel_dir, f) if rel_dir != '.' else f
            all_files.append(rel_file)
            print(rel_file)
        break
    sys.exit()
