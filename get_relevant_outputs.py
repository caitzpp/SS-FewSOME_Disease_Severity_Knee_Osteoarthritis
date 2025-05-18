import os
import shutil
from ss_fewsome.utils import get_best_epoch

#TODO: make sure to run it in the SS-FewSOME_Disease_xxx dir using cd
mod_prefix= "mod_2"

best_epochs = 990
outputs_path = "./ss_fewsome/outputs"
save_path = "./ss_fewsome/downloads"
margin = None

folders = ['models'] #'results', 'dfs', 'models', 
stages = [ 'stage_severe_pred']  #'stage_severe_pred', 'stage3' , 'ss',

if __name__=="__main__":
    os.makedirs(save_path, exist_ok=True)

    # #TODO: Fix this formula
    # for stage in stages:
    #     if best_epochs is None:
    #         best_epochs = get_best_epoch(os.path.join('./ss_fewsome', 'outputs/logs/') + stage + '/', last_epoch = current_epoch, metric='ref_centre', model_prefix = mod_prefix)
    #         print(best_epochs)

    for folder in folders:
        temp_file_path = os.path.join(outputs_path, folder)
        temp_save_path = os.path.join(save_path, folder)
        if folder in ['logs']:
            for stage in stages:
                temp_file_path2 = os.path.join(temp_file_path, stage)
                temp_save_path2 = os.path.join(temp_save_path, stage)
                os.makedirs(temp_save_path2, exist_ok=True)
                files_total = os.listdir(temp_file_path2)
               
                files=[]
                for key in best_epochs[stage].keys():
                    ##if best_epochs[stage]
                    files = files + [file for file in files_total if ('seed_' + str(key) in file) & (mod_prefix in file) ]
                for file in files:
                    shutil.copy(os.path.join(temp_file_path2, file), temp_save_path2)
                    os.remove(os.path.join(temp_file_path2, file))
        else:
            for stage in stages:
                temp_file_path2 = os.path.join(temp_file_path, stage)
                temp_save_path2 = os.path.join(temp_save_path, stage)
                os.makedirs(temp_save_path2, exist_ok=True)
                files_total = os.listdir(temp_file_path2)
                if isinstance(best_epochs, dict):
                    files=[]
                    for key in best_epochs[stage].keys():
                        if margin is not None:
                            files = files + [file for file in files_total if (('epoch_' + str(best_epochs[key]) ) in file) & ('seed_' + str(key) in file) & (mod_prefix in file) & ('margin_' + margin in file) ]
                        else:
                            files = files + [file for file in files_total if (('epoch_' + str(best_epochs[key]) ) in file) & ('seed_' + str(key) in file) & (mod_prefix in file) ]
                else:
                     files = []
                     files = files + [file for file in files_total if (('epoch_' + str(best_epochs) ) in file) & (mod_prefix in file) ]
                for file in files:
                    shutil.copy(os.path.join(temp_file_path2, file), temp_save_path2)
                    os.remove(os.path.join(temp_file_path2, file))
        
        print(f"Folder {folder} scanned")
        # for file in files_total:
        #     if file not in files:
        #         os.remove(os.path.join(temp_file_path2, file))
    print("All files copied!")