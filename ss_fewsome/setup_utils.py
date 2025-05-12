import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    #what stages to perform
    parser.add_argument('--train_ss', type=int, default=1)
    parser.add_argument('--stage2', type=int, default = 1)
    parser.add_argument('--stage3', type=int, default = 1)
    parser.add_argument('--stage_severe_pred', type=int, default = 1)


    #the same for all models
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--train_ids_path', type=str, default='/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/meta/')
    parser.add_argument('--task', type=str, default='test')
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dry_run', action= 'store_true')
    parser.add_argument('--model_name', type=str, default='mod_1')
    parser.add_argument('--augmentations', type=str, default="crop, cutpaste")
    parser.add_argument('--normal_augs', type=str, default="sharp, bright, jitter")
    parser.add_argument('--seed', type=int, default=None)

    #path of directory
    parser.add_argument('--dir_path', type=str, default='/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome')


    #details of data if not running train_ss
    parser.add_argument('--stage1_path_to_anom_scores', type=str, default = '/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome/outputs/dfs/ss/')
    parser.add_argument('--stage_severe_path_to_anom_scores', type=str, default = '/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome/outputs/dfs/stage_severe_pred/')
    parser.add_argument('--stage2_path_to_anom_scores', type=str, default = '/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome/outputs/dfs/stage2/')
    parser.add_argument('--stage3_path_to_anom_scores', type=str, default = '/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome/outputs/dfs/stage3/')
    parser.add_argument('--stage1_path_to_logs', type=str, default = '/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome/outputs/logs/ss/')
    parser.add_argument('--stage_severe_path_to_logs', type=str, default = '/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome/outputs/logs/stage_severe_pred/')
    parser.add_argument('--stage2_path_to_logs', type=str, default = '/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome/outputs/logs/stage2/')
    parser.add_argument('--stage3_path_to_logs', type=str, default = '/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/ss_fewsome/outputs/logs/stage3/')


    #epochs and N for each stage
    parser.add_argument('--ss_N', type=int, default=30)
    parser.add_argument('--stage_severe_pred_N', type=int, default=30)
    parser.add_argument('--stage2_N', type=int, default=30)
    parser.add_argument('--ss_epochs', type=int, default=400)
    parser.add_argument('--stage2_epochs', type=int, default=1000)
    parser.add_argument('--stage3_epochs', type=int, default=1000)
    parser.add_argument('--stage_severe_pred_epochs', type=int, default=990)

    #patching parameters for ss stage
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--patchsize', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)

    #evaluate on test set or not
    parser.add_argument('--ss_test', type=int, default=1)
    parser.add_argument('--stage2_test', type=int, default = 1)
    parser.add_argument('--stage3_test', type=int, default=1)


    parser.add_argument('--save_models', type=int, default=0)
    parser.add_argument('--save_anomaly_scores', type=int, default=1)
    parser.add_argument('--meta_data_dir', type=str, default = '/home2/c.zuppinger/VT9_SSFewSOME/SS-FewSOME_Disease_Severity_Knee_Osteoarthritis/meta/kxr_sq_bu00.txt' )
    parser.add_argument('--get_oarsi_results', type=int, default = 0)

    parser.add_argument('--start_margin', type=float, default = 0.8)
    parser.add_argument('--severe_num_pseudo_labels', type=float, default = 3)
    parser.add_argument('--precision', type=float, default=0.0001)



    args = parser.parse_args()
    return args