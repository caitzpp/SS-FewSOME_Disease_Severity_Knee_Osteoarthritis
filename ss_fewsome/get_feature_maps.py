import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from model import ALEXNET_nomax_pre, vgg16
from setup_utils import parse_arguments
from load_utils import ImageFolderWithPaths
from utils import create_patches
from torchvision import transforms

# NEPOCH=990
# seeds= ['1001', '138647', '193', '34', '44'] #['1001', '138647', '193', '34', '44', '71530', '875688', '8765', '985772', '244959']
# BATCH_SIZE= 1
# patches = True
# stage = 'stage_severe_pred'
# on_test_set = False

NEPOCH=400
seeds= ['1001', '138647', '193', '34', '44', '71530', '875688', '8765', '985772', '244959']
BATCH_SIZE= 1
patches = True
stage = 'ss'
on_test_set = False


def extract_features(model, dataloader, device, batch_size = 1, save_path=None):
    model.eval()
    with torch.no_grad():
        for i, (data, labels, filenames) in enumerate(dataloader):
            img1 = data.to(device)
            features = model(img1.float())
            
            if patches: 
                features = create_patches(features, args.padding, args.patchsize, args.stride)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))[:, :, 0, 0].squeeze(1)  # shape: (N, C)
            
            filename = str(filenames[0])
            filename = filename.split('/')[-1]
            filename = filename.split('.')[0]

            label = str(labels[0].item())

            features = features.cpu()
      
            if save_path:
                temp_save_path = os.path.join(save_path, label)

                np.save(os.path.join(temp_save_path, filename + '.npy'), features[0].numpy())
        
if __name__=="__main__":
    try:
       args = parse_arguments()
       print(f"Arguments: {args}")
    except Exception as e:
       print("CRITICAL ERROR DURING ARGUMENT PARSING.")
       print(e)
       raise
    
    mod_prefix = args.model_name
    device = args.device
    data_path = args.data_path
    model_path = os.path.join(args.model_path, stage)
    models = os.listdir(model_path)

    for seed in seeds:
        if isinstance(NEPOCH, dict):
            print("Logic not built yet")
        else:
            if on_test_set:
                model_name = [f for f in models if (mod_prefix in f) & (f"seed_{seed}" in f) & ("on_test_set" in f) & (str(NEPOCH) in f)]
            else:
                model_name = [f for f in models if (mod_prefix in f) & (f"seed_{seed}" in f) & ("on_test_set" not in f) & (str(NEPOCH) in f)]
            if len(model_name)==1:
                model_name = str(model_name[0])
            else:
                print("More than one model_name: ", model_name)
            
        save_path = os.path.join(args.feature_save_path, stage, model_name)
        os.makedirs(save_path, exist_ok=True)
        scores = [0, 1, 2, 3, 4]


        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # match input size for AlexNet
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet normalization
            #                     std=[0.229, 0.224, 0.225]),
        ])

        #load data
        train_save_path = os.path.join(save_path, 'train')
        for i in range(len(scores)):
            os.makedirs(os.path.join(train_save_path, str(scores[i])), exist_ok=True)

        train_dataset = ImageFolderWithPaths(os.path.join(data_path, 'train'), transform=transform) #add train data path
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

        checkpoint = torch.load(os.path.join(model_path, model_name))

        if stage == 'ss':
            model = ALEXNET_nomax_pre().to(device)
        else:
            model = vgg16().to(args.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        #push data through model to get feature vectors

        extract_features(model, dataloader, 
                        device = device, 
                        batch_size=1, 
                        save_path=train_save_path)

        #load data
        test_save_path = os.path.join(save_path, 'test')
        for i in range(len(scores)):
            os.makedirs(os.path.join(test_save_path, str(scores[i])), exist_ok=True)

        test_dataset = ImageFolderWithPaths(os.path.join(data_path, 'test'), transform=transform) 
        dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        extract_features(model, dataloader, 
                        device = device, 
                        batch_size=1, 
                        save_path=test_save_path)