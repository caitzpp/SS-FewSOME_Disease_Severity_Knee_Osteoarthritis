import os
import shutil

#TODO: make sure to run it in the SS-FewSOME_Disease_xxx dir using cd
mod_prefix= "mod_2"

best_epochs = {'1001': 1990, '138647': 2110, '193': 1550, '34': 2040, '44': 2070} # for stage 2 of mod_2 {'1001': 1390, '138647': 1190, '193': 800, '34': 1270, '44': 1290, '71530': 1250, '875688': 800, '8765': 800, '985772': 1320, '244959': 1380}
outputs_path = "./ss_fewsome/outputs"
save_path = "./ss_fewsome/downloads"

folders = ['logs', 'results', 'dfs', 'models'] #'results', 'dfs', 'models', 
stages = [ 'stage3']  #'stage_severe_pred', 'stage3' , 'ss',

if __name__=="__main__":
    os.makedirs(save_path, exist_ok=True)

    for folder in folders:
        temp_file_path = os.path.join(outputs_path, folder)
        temp_save_path = os.path.join(save_path, folder)
        if folder in ['logs']:
            for stage in stages:
                temp_file_path2 = os.path.join(temp_file_path, stage)
                temp_save_path2 = os.path.join(temp_save_path, stage)
                os.makedirs(temp_save_path2, exist_ok=True)
                files_total = os.listdir(temp_file_path2)
                if isinstance(best_epochs, dict):
                    files=[]
                    for key in best_epochs.keys():
                        files = files + [file for file in files_total if ('seed_' + str(key) in file) & (mod_prefix in file) ]
                print(len(files))
                for file in files:
                    shutil.copy(os.path.join(temp_file_path2, file), temp_save_path2)
        else:
            for stage in stages:
                temp_file_path2 = os.path.join(temp_file_path, stage)
                temp_save_path2 = os.path.join(temp_save_path, stage)
                os.makedirs(temp_save_path2, exist_ok=True)
                files_total = os.listdir(temp_file_path2)
                if isinstance(best_epochs, dict):
                    files=[]
                    for key in best_epochs.keys():
                        files = files + [file for file in files_total if (('epoch_' + str(best_epochs[key]) ) in file) & ('seed_' + str(key) in file) & (mod_prefix in file) ]
                print(len(files))
                for file in files:
                    shutil.copy(os.path.join(temp_file_path2, file), temp_save_path2)
        
        print(f"Folder {folder} scanned")
    print("All files copied!")