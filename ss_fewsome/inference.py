import torch
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from datasets.oa_knee import oa
from torch.utils.data import DataLoader
from model import *
from evaluate import *
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support, f1_score
from utils import *
import torch.multiprocessing
from train import *
from setup_utils import parse_arguments
import sys
from dotenv import load_dotenv

torch.multiprocessing.set_sharing_strategy('file_system')

seed = None
TRAIN_PLATEAU_EPOCH = 400
shots = 0
semi=0
self_supervised = 1
patches = True



def inference(args, model_path, N, model_name, num_ss, patches, seed, shots, semi, self_supervised,lr = 1e-6, bs = 1,
           beta1 = 0.9, beta2 = 0.999, n_eps = 1e-08, weight_decay = 0.1,
           metric = 'centre_mean'):
    if args.task == 'all':
        model = ALEXNET_nomax_pre().to(args.device)
        dataset = oa(args.data_path, task = args.task)
        train_dataset =  oa(args.data_path, task='train', stage='ss', N = N, 
                            shots = shots, semi = semi, 
                            self_supervised = self_supervised, 
                            num_ss = num_ss, augmentations = args.augmentations, 
                            normal_augs = args.normal_augs, train_info_path = args.train_ids_path, 
                            seed = seed)
    else:
        print("Invalid task specified.")
        sys.exit()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                         betas=(beta1, beta2), eps=n_eps)
    train_indexes = list(range(0, train_dataset.__len__()))
    print(f"Length Train Dataset {train_dataset.__len__()}")
    criterion = ContrastiveLoss(args.device)
    print("Criterion loaded")

    if args.get_oarsi_results:
        df, results, ref_info,ref_std,oarsi_res = evaluate_severity(patches, args.padding,args.patchsize, 
                                                                    args.stride,seed, train_dataset, dataset,
                                                                      model, args.data_path, criterion, args.device, 
                                                                      shots, args.meta_data_dir, args.get_oarsi_results)
    else:
        df, results, ref_info,ref_std = evaluate_severity(patches, args.padding,args.patchsize, args.stride,seed, 
                                                          train_dataset, dataset, model, args.data_path, 
                                                          criterion, args.device, shots, args.meta_data_dir, 
                                                          args.get_oarsi_results)
    oas_test = []
    mid_test=[]
    mid_2_test=[]
    sevs_test =[]
    sps_test=[]

    oas_test.append(results.loc[metric, 'auc'])
    mid_test.append(results.loc[metric,'auc_mid'])
    mid_2_test.append(results.loc[metric,'auc_mid2'])
    sevs_test.append(results.loc[metric, 'auc_sev'])
    sps_test.append(results.loc[metric, 'spearman'])


    logs_df = pd.concat([pd.DataFrame(oas_test, columns=['OA>0']),  pd.DataFrame(mid_test, columns=['OA>1']),pd.DataFrame(mid_2_test, columns=['OA>2']), pd.DataFrame(sevs_test, columns=['OA>3']), pd.DataFrame(sps_test, columns=['spearman'])], axis =1)

    if args.get_oarsi_results:
        write_results_inference(df, results, model_name + '_all', logs_df, model, optimizer, args, oarsi_res)
    else:
        write_results_inference(df, results, model_name + '_all', logs_df, model, optimizer,  args)


if __name__ == '__main__':
    print("Starting script")
    sys.stdout.flush()

    try:
        args = parse_arguments()
        print(f"Arguments: {args}")
        sys.stdout.flush()
    except Exception as e:
        print("CRITICAL ERROR DURING ARGUMENT PARSING.")
        print(e)
        sys.stdout.flush()
        raise

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(1001)
    torch.cuda.manual_seed(1001)
    torch.cuda.manual_seed_all(1001)


    base_output_dir = os.path.join(args.dir_path, 'outputs')
    subdirs = [
        '',  # outputs/
        'label_details_test',
        'results_test',
        'dfs_test',
        'models_test',
        'logs_test',
        'oarsi_test'
    ]

    # Create main output folders
    for subdir in subdirs:
        path = os.path.join(base_output_dir, subdir)
        os.makedirs(path, exist_ok=True)

    # Stage-specific subfolders
    stages = ['ss']
    stage_subdirs = ['results_test', 'dfs_test', 'models_test', 'logs_test', 'oarsi_test', 'label_details_test']

    for stage in stages:
        for subdir in stage_subdirs:
            path = os.path.join(base_output_dir, subdir, stage)
            os.makedirs(path, exist_ok=True)

    model_path = os.path.join(args.dir_path, 'outputs/models/ss/')
    
    if seed == None:
        seeds =[1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
    else:
        seeds =[seed]

    N = args.ss_N
    num_ss = args.ss_N

    for seed in seeds:
        model_name = f'ss_training_{args.model_name}_bs_{args.bs}_task_test_lr_{str(args.lr)}_N_30_seed_{seed}_epoch_{TRAIN_PLATEAU_EPOCH}'
        model_p = os.path.join(model_path, model_name)

        checkpoint = torch.load(model_p, map_location=args.device)
        print(checkpoint.keys())
        break
        # inference(args, model_path,N, model_name, num_ss, patches, seed, shots, semi, self_supervised, num_ss, seed,lr = 1e-6, bs = 1,
        #    beta1 = 0.9, beta2 = 0.999, n_eps = 1e-08, weight_decay = 0.1,
        #    metric = 'centre_mean')
