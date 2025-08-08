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
import wandb
from dotenv import load_dotenv

torch.multiprocessing.set_sharing_strategy('file_system')

TRAIN_PLATEAU_EPOCH = 400
SEVERE_PRED_EPOCH = 990

def ss_training(args, model_temp_name_ss, N, epochs, num_ss, shots, self_supervised, semi, seed = None, eval_epoch = 1): #trains the model and evaluates every 10 epochs for all seeds OR trains the model for a specific number of epochs for specified seed
  use_wandb = args.use_wandb
  print("Trying to load val_dataset")
  sys.stdout.flush()
  val_dataset =  oa(args.data_path, task = 'test_on_train', train_info_path = args.train_ids_path)
  print("Val Dataset Loaded")
  sys.stdout.flush()
  if args.ss_test:
      test_dataset =  oa(args.data_path, task = args.task)
      #print("ss_test working, test_dataset loaded")
  else:
      test_dataset = None

  if seed == None:
     seeds =[1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
  else:
      seeds =[seed]

  current_epoch = 0
  for seed in seeds:
      model = ALEXNET_nomax_pre().to(args.device)
      train_dataset =  oa(args.data_path, task='train', stage='ss', N = N, shots = shots, semi = semi, self_supervised = self_supervised, num_ss = num_ss, augmentations = args.augmentations, normal_augs = args.normal_augs, train_info_path = args.train_ids_path, seed = seed)
      print(f"Training with {len(train_dataset)} samples")
      #print(f"First 5 paths:" {train_dataset.paths[:5]})
      if use_wandb == 1:
            run = wandb.init(project="SS-Fewsome", name=model_name_temp_ss + '_seed_' + str(seed), config= {
                #     "epochs": epochs,
                #     "seed": seed,
                #     "model_name": model_name_temp_ss + '_seed_' + str(seed),
                #     "N": N,
                #     "shots": shots,
                #     "patches": True,
                #     "eval_epoch": eval_epoch,
                #     "lr": args.lr,
                #     "bs": args.bs,
                #     "weight_decay": 0.1,
                #     "metric": 'centre_mean',
                # # "model_parameters": model.parameters(),
                })
            wandb_config = wandb.config
            train(train_dataset, val_dataset, N, model, epochs, seed, eval_epoch, shots, 
                  model_name = wandb_config.model_name, args=args, 
                  current_epoch=current_epoch, metric=wandb_config.metric, 
                  patches =wandb_config.patches, test_dataset = test_dataset, 
                  use_wandb=use_wandb,
                  weight_decay=wandb_config.weight_decay
                   , lr = wandb_config.lr, bs = wandb_config.bs, beta1 = wandb_config.beta1,
                    beta2 = wandb_config.beta2,
                    eps = wandb_config.eps)
      else:
            train(train_dataset, val_dataset, N, model, epochs, seed, eval_epoch, shots, model_name_temp_ss + '_seed_' + str(seed), args, current_epoch, metric='centre_mean', patches =True, test_dataset = test_dataset, use_wandb=use_wandb )
      print("Training Done")
      del model
      print("Model Deleted")
      sys.stdout.flush()

  return os.path.join(args.dir_path, 'outputs/dfs/ss/'), os.path.join(args.dir_path, 'outputs/logs/ss/')




def dclr_training(args, model_temp_name_stage, stage, pseudo_label_ids, epochs, num_ss, current_epoch, model_prefix, self_supervised = 1, semi= 0, seed=None):

    val_dataset =  oa(args.data_path, task = 'test_on_train', train_info_path = args.train_ids_path)
    if args.stage3_test:
        test_dataset =  oa(args.data_path, task = args.task)
    else:
        test_dataset = None

    if seed is not None:
        seed = [seed]
    elif stage == 'stage2':
        seeds =[1001, 138647, 193, 34, 44, 71530, 875688, 8765, 985772, 244959]
        #print(f"Using seeds {seeds}")
    else:
        seeds =[ 1001, 138647, 193, 34, 44]
    for seed in seeds:
        model = vgg16().to(args.device)
        train_dataset =  oa(args.data_path, task='train', stage= stage, semi = semi, self_supervised = self_supervised, num_ss = num_ss, augmentations = args.augmentations, normal_augs = args.normal_augs, train_info_path = args.train_ids_path, seed = seed, pseudo_label_ids = pseudo_label_ids)
        N = train_dataset.N
        shots = train_dataset.shots
        if isinstance(current_epoch, dict):
            ep = current_epoch[str(seed)]
        else:
            ep=current_epoch
        train(train_dataset, val_dataset, N, model, epochs, seed, args.eval_epoch, shots, model_temp_name_stage + '_seed_' + str(seed) +  '_N_' + str(N), args, ep, metric='w_centre', patches = False, test_dataset = test_dataset )
        del model
    current_epoch = get_best_epoch(os.path.join(args.dir_path, 'outputs/logs/') + stage + '/', last_epoch = current_epoch, metric='ref_centre', model_prefix = model_prefix)

    return current_epoch, os.path.join(args.dir_path, 'outputs/dfs/') + stage + '/'



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
  if args.dry_run:
       print("Dry run: loading small batches and doing forward pass...")
       model = ALEXNET_nomax_pre().to(args.device)

       args.bs = 1
       index = 0
       seed_temp = 123

       train_set = oa(args.data_path, task='train', stage='ss', N = args.ss_N, shots = 0, semi = 0, self_supervised = 1, num_ss = args.ss_N, augmentations = args.augmentations, normal_augs = args.normal_augs, train_info_path = args.train_ids_path, seed = 1001)
       img1, img2, labels, base, _, _  = train_set.__getitem__(index, seed_temp)

       # Forward Loop
       img1 = img1.to(args.device)
       img2 = img2.to(args.device)

       labels = labels.to(args.device)

       with torch.no_grad():
            output1 = model.forward(img1.float())
            output2 = model.forward(img2.float())
       print(f"Forward pass shape - Output 1: {output1.shape}\n Output 2: {output2.shape}")
       exit()

  if args.train_ss:
      if args.use_wandb == 1:
          dotenv_path = os.path.join(args.dir_path, '.env')
          load_dotenv(dotenv_path)
          wandb.login(key=os.getenv("WANDB_API_KEY"))
      print("Trying self-supervised Training")
      sys.stdout.flush()
      model_name_temp_ss = stages[0] + '/' + 'ss_training_' + model_name_temp  + '_N_' + str(args.ss_N)
      print(f"Model Name for temp SS: {model_name_temp_ss}")
      sys.stdout.flush()
      stage1_path_to_anom_scores, stage1_path_to_logs = ss_training(args, model_name_temp_ss, N=args.ss_N, epochs = TRAIN_PLATEAU_EPOCH, num_ss = args.ss_N, shots=0, self_supervised=1, semi=0, seed = None, eval_epoch = args.eval_epoch)
  else:
      print("Using anomaly scores & logs")
      sys.stdout.flush()
      stage1_path_to_anom_scores = args.stage1_path_to_anom_scores
      stage1_path_to_logs = args.stage1_path_to_logs


  print_ensemble_results(stage1_path_to_anom_scores, TRAIN_PLATEAU_EPOCH, stages[0], 'centre_mean', args.meta_data_dir, args.get_oarsi_results, model_name_prefix = args.model_name)
  sys.stdout.flush()
  #stage2 is to DCLR-FewSOME_OA ITER 1
  if args.stage2:
      pseudo_label_ids, margin = get_pseudo_labels(args.train_ids_path, stage1_path_to_anom_scores, args.data_path, margin = args.start_margin, metric = 'centre_mean', current_epoch=TRAIN_PLATEAU_EPOCH, num_pseudo_labels=args.stage2_N, model_name_prefix = args.model_name, model_name=stages[2] + '/' + model_name_temp, args= args)
      shots = len(pseudo_label_ids)
      model_name_temp_stage2 =  stages[2] + '/' + 'stage2_' + 'margin_' + str(margin) + '_' + model_name_temp + '_shots_' + str(shots)  + '_N_' + str(args.stage2_N)
      pd.DataFrame(pseudo_label_ids).to_csv(os.path.join(args.dir_path,'outputs/label_details/') + model_name_temp_stage2 + 'dclr_fewsome_OA_iter1_pseudo_anom_labels.csv')
      current_epoch, stage2_path_to_anom_scores = dclr_training(args, model_name_temp_stage2, stages[2], pseudo_label_ids= pseudo_label_ids, epochs = args.stage2_epochs, current_epoch = TRAIN_PLATEAU_EPOCH, model_prefix = args.model_name, num_ss=0, self_supervised = 0, semi= 1)
      if args.eval_epoch == 0:
           for key in current_epoch.keys():
               _,_ = dclr_training(args, model_name_temp_stage2, stages[2], pseudo_label_ids= pseudo_label_ids, epochs = current_epoch[key] - TRAIN_PLATEAU_EPOCH, current_epoch = TRAIN_PLATEAU_EPOCH, model_prefix = args.model_name, num_ss=0, self_supervised = 0, semi= 1, seed=int(key))

  else:
      stage2_path_to_anom_scores = args.stage2_path_to_anom_scores
      stage2_path_to_logs = args.stage2_path_to_logs
      current_epoch = get_best_epoch(args.stage2_path_to_logs, last_epoch = TRAIN_PLATEAU_EPOCH, metric='ref_centre', model_prefix = args.model_name)


  stage2_epoch = current_epoch
  print_ensemble_results(stage2_path_to_anom_scores, current_epoch, stages[2], 'w_centre', args.meta_data_dir, args.get_oarsi_results, model_name_prefix = args.model_name)
  print(current_epoch)
  #stage3 is to DCLR-FewSOME_OA ITER 2
  if args.stage3:
        pseudo_label_ids, margin =  get_pseudo_labels(args.train_ids_path, stage2_path_to_anom_scores, args.data_path, margin = args.start_margin, metric = 'w_centre', current_epoch=current_epoch, num_pseudo_labels=args.stage3_num_pseudo_labels, model_name_prefix = args.model_name, model_name=stages[3] + '/' + model_name_temp, args=args)
        shots = len(pseudo_label_ids)
        model_name_temp_stage3 =  stages[3] + '/' +  'stage3_' + 'margin_' + str(margin) + '_' + model_name_temp + '_shots_' + str(shots)
        pd.DataFrame(pseudo_label_ids).to_csv(os.path.join(args.dir_path,'outputs/label_details/') + model_name_temp_stage3 + 'dclr_fewsome_OA_iter2_pseudo_anom_labels.csv')
        current_epoch, stage3_path_to_anom_scores = dclr_training(args, model_name_temp_stage3, stages[3], pseudo_label_ids = pseudo_label_ids, epochs = args.stage3_epochs, num_ss=0, current_epoch = current_epoch, model_prefix = args.model_name, self_supervised = 0, semi= 1)
        if args.eval_epoch == 0:
              for key in current_epoch.keys():
                  _,_ = dclr_training(args, model_name_temp_stage3, stages[3], pseudo_label_ids = pseudo_label_ids, epochs = current_epoch[key] -  stage2_epoch[key], current_epoch = stage2_epoch[key], model_prefix = args.model_name, num_ss=0, self_supervised = 0, semi= 1, seed=int(key))
  else:
      stage3_path_to_anom_scores = args.stage3_path_to_anom_scores
      current_epoch = get_best_epoch(args.stage3_path_to_logs, last_epoch = current_epoch, metric='ref_centre', model_prefix = args.model_name)

  stage3_epoch = current_epoch
  print(current_epoch)
  print_ensemble_results(stage3_path_to_anom_scores, current_epoch, stages[3], 'w_centre', args.meta_data_dir, args.get_oarsi_results, model_name_prefix = args.model_name)

  if args.stage_severe_pred:
      pseudo_label_ids, severe_margin = get_pseudo_labels(args.train_ids_path, stage1_path_to_anom_scores, args.data_path, margin = args.start_margin, metric = 'centre_mean', current_epoch=TRAIN_PLATEAU_EPOCH, num_pseudo_labels=args.severe_num_pseudo_labels, model_name_prefix = args.model_name, model_name=stages[1] + '/' + model_name_temp, args = args)
      shots = len(pseudo_label_ids)
      model_name_temp_sev =  stages[1] + '/' + 'stage_sev_pred_' + 'margin_' + str(severe_margin) + '_' + model_name_temp + '_shots_' + str(shots)  + '_N_' + str(args.stage_severe_pred_N)
      pd.DataFrame(pseudo_label_ids).to_csv(os.path.join(args.dir_path,'outputs/label_details/') + model_name_temp_sev + 'dclr_fewsome_sev_pseudo_anom_labels.csv')
      current_epoch, stage_severe_path_to_anom_scores = dclr_training(args, model_name_temp_sev, stages[1], pseudo_label_ids= pseudo_label_ids, epochs = args.stage_severe_pred_epochs, current_epoch = TRAIN_PLATEAU_EPOCH, model_prefix = args.model_name, num_ss=0, self_supervised = 0, semi= 1)
      if args.eval_epoch == 0:
           for key in current_epoch.keys():
               _,_ = dclr_training(args,model_name_temp_sev, stages[1], pseudo_label_ids= pseudo_label_ids, epochs = current_epoch[key] - TRAIN_PLATEAU_EPOCH, current_epoch = TRAIN_PLATEAU_EPOCH, model_prefix = args.model_name, num_ss=0, self_supervised = 0, semi= 1)

  else:
      stage_severe_path_to_anom_scores = args.stage_severe_path_to_anom_scores
      current_epoch = get_best_epoch(args.stage_severe_path_to_logs, last_epoch = TRAIN_PLATEAU_EPOCH, metric='ref_centre', model_prefix = args.model_name)

  stage_severe_epoch = SEVERE_PRED_EPOCH
  print_ensemble_results(stage_severe_path_to_anom_scores, SEVERE_PRED_EPOCH, stages[1], 'w_centre', args.meta_data_dir, args.get_oarsi_results, model_name_prefix = args.model_name)


  #rerun results for all stages
  print_ensemble_results(stage1_path_to_anom_scores, TRAIN_PLATEAU_EPOCH, stages[0], 'centre_mean', args.meta_data_dir, args.get_oarsi_results, model_name_prefix = args.model_name)
  print_ensemble_results(stage2_path_to_anom_scores, stage2_epoch, stages[2], 'w_centre', args.meta_data_dir, args.get_oarsi_results, model_name_prefix = args.model_name)
  print_ensemble_results(stage3_path_to_anom_scores, stage3_epoch, stages[3], 'w_centre', args.meta_data_dir, args.get_oarsi_results, model_name_prefix = args.model_name)
  print_ensemble_results(stage_severe_path_to_anom_scores, SEVERE_PRED_EPOCH, stages[1], 'w_centre', args.meta_data_dir, args.get_oarsi_results, model_name_prefix = args.model_name)
  combine_results(stage3_path_to_anom_scores, stage_severe_path_to_anom_scores, stage3_epoch, stage_severe_epoch, 'w_centre', args.meta_data_dir, args.get_oarsi_results, model_name_prefix = args.model_name)
