import torch.utils.data as data
from PIL import Image
import torch
import os
import numpy as np
import random
import pandas as pd
import nibabel as nib
from transforms import *


class oa(data.Dataset):



    def __init__(self, root: str,
            task, stage='ss', N=0, shots=0, semi=0, self_supervised=0, num_ss=0, augmentations='', normal_augs ='', train_info_path = '', seed= 0, pseudo_label_ids=None, norm_pseudo_label=None):
        super().__init__()
        self.paths = []
        self.targets=[]
        self.root_dir = root
        self.task = task
        self.N =N
        self.targets2=[]
        self.paths2=[]
        self.shots=shots
        self.semi=semi
        self.self_supervised = self_supervised
        self.num_ss = num_ss
        self.augmentations=augmentations
        self.normal_augs = normal_augs

        self.augmentations = self.augmentations.split(', ')
        self.normal_augs=self.normal_augs.split(', ')

        if (task =='train') :


            files = os.listdir(os.path.join(root, 'train/0'))
            if (stage == 'ss') | (stage=='stage2') | (stage=='stage_severe_pred'):
                seed_file = 'train_seed_' + str(seed) +'.csv'
                train_ids = pd.read_csv(os.path.join(train_info_path, seed_file)).iloc[:,2].tolist()
                self.paths = self.paths + [os.path.join(root, 'train', ids) for ids in train_ids]
                self.targets = self.targets + ([0]*len(train_ids))
                self.paths2 = self.paths2 + [os.path.join(root, 'train', ids) for ids in train_ids]
                self.targets2 = self.targets2 + ([0]*len(train_ids))

            elif stage =='stage3':
                train_ids = pd.read_csv(os.path.join(train_info_path, 'train_ids.csv')).iloc[:,1].tolist()
                self.paths = self.paths + [os.path.join(root, 'train', ids) for ids in train_ids]
                self.targets = self.targets + ([0]*len(train_ids))
                self.paths2 = self.paths2 + [os.path.join(root, 'train', ids) for ids in train_ids]
                self.targets2 = self.targets2 + ([0]*len(train_ids))


            if pseudo_label_ids is not None:
                for pseudo in pseudo_label_ids:
                 if pseudo not in self.paths2:
                    self.paths2.append( pseudo)
                    self.targets2.append(5)

            if norm_pseudo_label is not None:
                    for pseudo in norm_pseudo_label:
                      if pseudo not in self.paths2:
                        self.paths.append( pseudo)
                        self.targets.append(0)
                        self.paths2.append( pseudo)
                        self.targets2.append(0)


            self.N = len(self.paths)
            self.shots = len(self.paths2) - len(self.paths)

            print('The stage is {}'.format(stage))
            print('The number of normal training instances is {}'.format(self.N))
            print('The number of anomalous instances are {}'.format(self.shots))


        elif task == 'test_on_train':
            files = os.listdir(os.path.join(root, 'train/0'))
            folders = ['0','1','2','3','4']
            for folder in folders:
                    val_files = os.listdir(os.path.join(root, 'train', folder))
                    for f in val_files:
                         self.paths.append(os.path.join(root, 'train', folder, f))
                         self.targets.append(int(folder))

        elif task == 'validation':
            folders = os.listdir(os.path.join(root, 'val'))
            for folder in folders:
                    val_files = os.listdir(os.path.join(root, 'val',folder))
                    for f in val_files:
                        self.paths.append(os.path.join(root,'val', folder, f))
                        self.targets.append(int(folder))

        elif task == 'test':
            folders = os.listdir(os.path.join(root, 'test'))
            for folder in folders:
                    val_files = os.listdir(os.path.join(root, 'test', folder))
                    for f in val_files:
                        self.paths.append(os.path.join(root, 'test', folder, f))
                        self.targets.append(int(folder))




        self.targets=np.array(self.targets)




    def __len__(self):

        return len(self.paths)



    def __getitem__(self, index: int, seed = 1, base_ind=-1):

        base=False

        if index >= len(self.paths): #shots
            target = torch.FloatTensor([self.targets2[index]])
            try:
                img = Image.open(self.paths2[index]).resize((224, 224), Image.BILINEAR)
                #img = Image.open(self.paths2[index])
            except Exception as e:
                print(f"[ERROR] Failed to load image at index {index}: {self.paths2[index]}")
                raise e
            img = torch.FloatTensor(np.asarray(img ).copy() ) / 255
            file=self.paths2[index]

        else:
            target = torch.FloatTensor([self.targets[index]])
            try:
                img = Image.open(self.paths[index]).resize((224, 224), Image.BILINEAR)

                #img = Image.open(self.paths[index])
            except Exception as e:
                print(f"[ERROR] Failed to load image at index {index}: {self.paths[index]}")
                raise e
            img = torch.FloatTensor(np.asarray(img ).copy() ) / 255
            file=self.paths[index]



        orig_label = target
        img = torch.stack((img,img,img),0)


        if self.task == 'train':
            np.random.seed(seed)

            ind = np.random.randint(4)
            if ind ==0:
                img = transform_function(self.normal_augs,  img)



            if self.semi ==1:
                ind_temp=np.random.randint(2)
                if ind_temp == 0:
                    ind = random.sample(range(0, (len(self.targets)) ), 1)[0]
                else:

                    ind = random.sample(range( (len(self.targets)), len(self.targets)+ self.shots ), 1)[0]

            else:
                ind = random.sample(range(0, len(self.targets)+self.num_ss), 1)[0]


            c=1
            while (ind == index):
                np.random.seed(seed * c)
                if self.semi ==1:

                    ind_temp=np.random.randint(2)
                    if ind_temp == 0:
                        ind = random.sample(range(0, (len(self.targets)) ), 1)[0]
                    else:
                        ind = random.sample(range( (len(self.targets)), len(self.targets)+ self.shots ), 1)[0]
                else:
                    ind = random.sample(range(0, len(self.targets)+self.num_ss), 1)[0]

                c=c+1

            if ind == base_ind:
              base = True


            if self.semi ==1 :
                    target2 = torch.FloatTensor([self.targets2[ind]])
                    try:
                        img2 = Image.open(self.paths2[ind]).resize((224, 224), Image.BILINEAR)

                        #img2 = Image.open(self.paths2[ind])
                    except Exception as e:
                        print(f"[ERROR] Failed to load image at index {index}: {self.paths2[ind]}")
                        raise e
                    img2 = torch.FloatTensor(np.asarray ( img2 ).copy() ) / 255
                    img2 = torch.stack((img2,img2,img2),0)

            else:
                if ind >= len(self.targets):
                    target2 = torch.FloatTensor([1])
                    i2=random.sample(range(0, len(self.targets)), 1)[0]
                    try:
                        img2 = Image.open(self.paths[i2]).resize((224, 224), Image.BILINEAR)

                        #img2 = Image.open(self.paths[i2])
                    except Exception as e:
                        print(f"[ERROR] Failed to load image at index {index}: {self.paths[i2]}")
                        raise e
                    img2 = np.asarray ( img2).copy()
                    img2 = transform_function(self.augmentations, img2)

                else:
                    target2 = torch.FloatTensor([self.targets[ind]])
                    try:
                        img2 = Image.open(self.paths[ind]).resize((224, 224), Image.BILINEAR)

                        #img2 = Image.open(self.paths[ind])
                    except Exception as e:
                        print(f"[ERROR] Failed to load image at index {index}: {self.paths[ind]}")
                        raise e
                    img2 = torch.FloatTensor(np.asarray ( img2 ).copy() ) / 255
                    img2 = torch.stack((img2,img2,img2),0)

            if isinstance(img2, tuple):
                img2, target = img2
                target = target.flatten()
            else:
                if target != target2:
                    target = torch.FloatTensor([1])
                else:
                    target = torch.FloatTensor([0])



        else:
            img2 = torch.Tensor([1])



        return img, img2, target, base,file, orig_label
