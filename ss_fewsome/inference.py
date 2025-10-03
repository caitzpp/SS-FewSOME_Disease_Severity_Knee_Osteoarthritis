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