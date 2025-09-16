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

TRAIN_PLATEAU_EPOCH = 400
SEVERE_PRED_EPOCH = 990

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
        'label_details',
        'results',
        'dfs',
        'models',
        'logs',
        'oarsi'
    ]

    # Create main output folders
    for subdir in subdirs:
        path = os.path.join(base_output_dir, subdir)
        os.makedirs(path, exist_ok=True)

    # Stage-specific subfolders
    stages = ['ss', 'stage_severe_pred', 'stage2', 'stage3']
    stage_subdirs = ['results', 'dfs', 'models', 'logs', 'oarsi', 'label_details']

    for stage in stages:
        for subdir in stage_subdirs:
            path = os.path.join(base_output_dir, subdir, stage)
            os.makedirs(path, exist_ok=True)

    model_name_temp = args.model_name  + '_bs_' + str(args.bs) + '_task_' + str(args.task)+  '_lr_' + str(args.lr)
    print(f"Temp Model Name: {model_name_temp}")
    sys.stdout.flush()
    if args.train_ss:
        print("Trying self-supervised Training")
        sys.stdout.flush()
        model_name_temp_ss = stages[0] + '/' + 'ss_training_' + model_name_temp  + '_N_' + str(args.ss_N)
        print(f"Model Name for temp SS: {model_name_temp_ss}")
        sys.stdout.flush()


    seeds =[1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]

    current_epoch = 0
    for seed in seeds:
        model = ALEXNET_nomax_pre().to(args.device)
        train_dataset =  oa(args.data_path, task='train', stage='ss', N = args.ss_N, shots = 0, semi = 0, self_supervised = 1, num_ss = args.ss_N, augmentations = args.augmentations, 
                            normal_augs = args.normal_augs, train_info_path = args.train_ids_path, 
                            seed = seed,
                            use_same_image=args.use_same_image)
        print(f"Training with {len(train_dataset)} samples")
        
        train_indexes = list(range(0, train_dataset.__len__()))

        for epoch in range(TRAIN_PLATEAU_EPOCH+1):
            print(f"Beginning with epoch: {epoch}")


            train_preds = []
            train_labels=[]
            loss_sum = 0
            #print("Starting epoch " + str(epoch+1))
            np.random.seed(epoch*seed)
            np.random.shuffle(train_indexes)

            batches = list(create_batches(train_indexes, args.bs))

            for batch_ind in range(len(batches)):
                

                iterations=0
                for inbatch_ind,index in enumerate(batches[batch_ind]):
                    
                    model.train()
                    iterations+=1
                    seed_temp = (epoch+1) * (inbatch_ind+1) * (batch_ind+1)

                    img1, img2, labels, base,_,_ = train_dataset.__getitem__(index, seed_temp)
            break
        break