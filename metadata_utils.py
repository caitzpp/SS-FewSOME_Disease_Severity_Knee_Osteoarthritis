import os
import random
import csv
import sys
import pandas as pd

def select_random_files_to_csv(input_dir, output_path, n = 150, seed = 1001):
    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            file_name = '0/'+str(f)
            #print(file_name)

            all_files.append(file_name)
    
    if seed is not None:
        random.seed(seed)
    
    selected_files = random.sample(all_files, min(n, len(all_files)))

    output_csv = os.path.join(output_path, "train_ids.csv")
    try:
        with open(output_csv, 'x', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['', 'id']) 
            for i, file_path in enumerate(selected_files):
                writer.writerow([i, file_path])
    except FileExistsError:
        print("File already exists!")

    print(f"Saved {len(selected_files)} files to {output_csv}")

def create_random_seed_files_to_csv(input_csv, seeds, output_dir, sample_n=30):
    df = pd.read_csv(input_csv)
    
    for seed in seeds:
        random.seed(seed)
        sampled_df = df.sample(n=min(sample_n, len(df)), random_state=seed).reset_index(drop=True)

        formatted_df = pd.DataFrame({
            '': sampled_df.index,
            'ind': sampled_df.index,
            'id': sampled_df['id']
        })
        
        output_path = os.path.join(output_dir, f'train_seed_{seed}.csv')
        formatted_df.to_csv(output_path, index=False)

        print(f"Saved seed: {seed}")


 