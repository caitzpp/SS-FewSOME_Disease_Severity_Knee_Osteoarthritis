import os
from metadata_utils import select_random_files_to_csv, create_random_seed_files_to_csv


DATA_PATH="../data/raw/train/0"
SAVE_PATH= "./SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/meta2"

# #TODO: get 150 samples from folder 0 and call train_ids.csv


# #TOD0: Using seeds xx get 30 samples, name train_seed_<seed>.csv
seeds = [34, 44, 193, 1001, 8765, 71530, 138647, 244959, 875688, 985772]

# #TODO: Save in meta2. Be sure to add this folder to .gitignore!!!

if __name__=="__main__":
    os.makedirs(SAVE_PATH, exist_ok=True)

    select_random_files_to_csv(DATA_PATH, SAVE_PATH, n = 150)

    create_random_seed_files_to_csv(os.path.join(SAVE_PATH, "train_ids.csv"), seeds, SAVE_PATH, sample_n=30)




  