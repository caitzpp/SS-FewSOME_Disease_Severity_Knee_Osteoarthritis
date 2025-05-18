import torch
import os
from torch.utils.data import DataLoader
from model import ALEXNET_nomax_pre
from setup_utils import parse_arguments
from load_utils import ImageFolderWithPaths
from utils import create_patches
from torchvision import transforms

NEPOCH=400
seeds=[]
BATCH_SIZE= 1
patches = True

def extract_features(model, dataloader, device, batch_size = 1, save_path=None):
    model.eval()

    all_features = []
    all_labels = []
    all_ids = []

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

            print(str(labels))

            if save_path:
                temp_save_path = os.path.join()

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            all_ids.append(filename)

            print(all_labels)
            print(all_ids)
            break
            # if isinstance(data, tuple):
            #     print(len(data))
            #     break
            # break
            #     if len(data)==6:
            #         img1, img2, labels, base, _, _ = data
            #     elif len(data)==3:
            #         img1, labels, base = data
            #         print(base)
            #     else: 
            #         raise ValueError("Unexpected dataset item format.")
            # else:
            #     img1 = data
            #     labels = None
            # img1 = img1.to(device)
            # features = model(img1.float()) #N, C, H, W

            # if patches:
            #     features = create_patches(features, args.padding, args.patchsize, args.stride)
            #     features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))[:, :, 0, 0].squeeze(1)  # shape: (N, C)

            # all_features.append(features.cpu())
            # all_labels.append(labels.cpu())
            # all_ids.extend(base)

    # Stack into tensors
    # features_tensor = torch.cat(all_features, dim=0)
    # labels_tensor = torch.cat(all_labels, dim=0)

    # return features_tensor, labels_tensor
            
        

if __name__=="__main__":
    try:
       args = parse_arguments()
       print(f"Arguments: {args}")
    except Exception as e:
       print("CRITICAL ERROR DURING ARGUMENT PARSING.")
       print(e)
       raise
    
    device = args.device
    data_path = args.data_path

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # match input size for AlexNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet normalization
                            std=[0.229, 0.224, 0.225]),
    ])

    #load data
    train_dataset = ImageFolderWithPaths(os.path.join(data_path, 'train'), transform=transform) #add train data path
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)


    #load model
    ## set seeds?

    model = ALEXNET_nomax_pre().to(device)

    #push data through model to get feature vectors

    extract_features(model, dataloader, 
                     device = device, 
                     batch_size=1, 
                     save_path=None)

    #patches?


    #final embedding


    # save features