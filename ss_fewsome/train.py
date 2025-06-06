import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
from evaluate import *
from utils import *
from sklearn.metrics import roc_curve, auc
import sys
import os


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, device):
        super(ContrastiveLoss, self).__init__()

        self.device=device


    def forward(self, output1, output2,  label):
        '''
         returns the loss based on one minus the cosine similarity between vectors output1 and output2 and the label
        '''

        sim = torch.nn.CosineSimilarity()(output1, output2).to(self.device)
        # print(f"Cosine Similarity: {sim}")
         
        dist = 1-sim
        dist = torch.clip(dist, min=0,max=1)
        # print(f"Distance: {dist}")
         
        loss_contrastive = torch.nn.BCELoss()(dist, label)
        # print(f"Contrastive Loss: {loss_contrastive}")
         
        assert loss_contrastive.requires_grad == True
        return loss_contrastive

def create_batches(lst, n):

    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def centre_sim(padding, patchsize, stride, ref_dataset, model, dev, shots):
        '''
            returns the similarity between the centre of normality and anomalous centre
        '''
        model.eval()
        mat, mat_anom = create_mat(ref_dataset, model, padding, patchsize, stride, dev, shots)
        c= create_centre(mat) #1 x 256
        c_anom = create_centre(mat_anom)
        return F.cosine_similarity(c, c_anom, dim=0).cpu().numpy()


def train(train_dataset, val_dataset, N, model, epochs, seed, eval_epoch, shots, model_name, args, current_epoch=0, metric='centre_mean', patches = True, test_dataset = None):

  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
  train_indexes = list(range(0, train_dataset.__len__()))
  print(f"Length Train Dataset {train_dataset.__len__()}")
  criterion = ContrastiveLoss(args.device)
  print("Criterion loaded")


  train_losses = []
  train_aucs=[]

  oas = []
  mid=[]
  mid_2=[]
  sevs =[]
  sps=[]
  trs=[]
  ref_max=[]
  ref_min=[]
  ref_sum=[]
  ref_mean=[]
  ref_var=[]
  ref_c=[]
  eps=[]
  channel_std=[]


  oas_test = []
  mid_test=[]
  mid_2_test=[]
  sevs_test =[]
  sps_test=[]




  for epoch in range(epochs+1):
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

              # Forward
              img1 = img1.to(args.device)
              img2 = img2.to(args.device)
              labels = labels.to(args.device)

              train_labels.append(torch.max(labels).detach().cpu().numpy())
              output1 = model.forward(img1.float())
              output2 = model.forward(img2.float())
            #   print(f"Output 1 Shape: {output1.shape}")
            #   print(f"Output 2 Shape: {output2.shape}")
               


              if patches:
                #   print(f"Creating patches")
                   
                  output1 = create_patches(output1, args.padding,args.patchsize, args.stride)
                  output1=F.adaptive_avg_pool2d(output1, (1,1) )[:,:,0,0].squeeze(1)
                  output2 = create_patches(output2, args.padding,args.patchsize, args.stride)
                  output2=F.adaptive_avg_pool2d(output2, (1,1) )[:,:,0,0].squeeze(1)

                  if not torch.all(torch.isfinite(output1)):
                      print("Model output 1 contains NaNs or Infs")
                       
                      continue
                  if not torch.all(torch.isfinite(output2)):
                      print("Model output2 contains NaNs or Infs")
                       
                      continue

                  if len(labels) == 1:
                      labels = torch.FloatTensor ([labels.item()]*output1.shape[0]).to(args.device)

              else:
                  if len(labels) != 1:
                      labels = torch.FloatTensor([torch.max(labels)]).to(args.device)

              if inbatch_ind ==0:
                  loss = criterion(output1,output2,labels)
              else:
                  loss = loss + criterion(output1,output2,labels)


              dist = 1-torch.nn.CosineSimilarity()(output1, output2)
              train_preds.append(torch.max(dist).detach().cpu().numpy())

           

          if not torch.isfinite(loss):
              print(f"Non-finite Loss detected")
               
              continue

          if loss.numel() == 1:
            loss_value = loss.item()
            loss_sum += loss_value / iterations
            # print(f"Loss Sum: {loss_sum}")
          else:
            # print("Loss tensor has more than one element — skipping loss.item()")
            continue

          # Backward and optimize
          optimizer.zero_grad()
        #   print("Zero grad done")
           

          loss.backward(retain_graph=True)
        #   try:
        #         with torch.autograd.set_detect_anomaly(True):
        #             loss.backward()  # Try without retain_graph
        #             print("Backward done")
                     
        #   except Exception as e:
        #         print("CRASH DURING BACKWARD:", e)
                 
        #         #torch.cuda.empty_cache()
        #         continue

          optimizer.step()

          torch.cuda.empty_cache()

         
      fpr, tpr, thresholds = roc_curve(train_labels ,train_preds)
      train_auc = auc(fpr, tpr)
      print('Train AUC is {}'.format(train_auc))
       
      train_losses.append((loss_sum / len(batches)))
      print("Epoch: {}, Train loss: {}".format(epoch+1, train_losses[-1]))
       


      if eval_epoch == 1:

          if (epoch % 10 == 0):
              if args.get_oarsi_results:
                    df, results, ref_info,ref_std,oarsi_res = evaluate_severity(patches, args.padding,args.patchsize, args.stride,seed, train_dataset, val_dataset, model, args.data_path, criterion, args.device, shots,  args.meta_data_dir, args.get_oarsi_results)
              else:
                    df, results, ref_info,ref_std = evaluate_severity(patches, args.padding,args.patchsize, args.stride,seed, train_dataset, val_dataset, model, args.data_path, criterion, args.device, shots,  args.meta_data_dir, args.get_oarsi_results)

              oas.append(results.loc[metric, 'auc'])
              mid.append(results.loc[metric,'auc_mid'])
              mid_2.append(results.loc[metric,'auc_mid2'])
              sevs.append(results.loc[metric, 'auc_sev'])
              sps.append(results.loc[metric, 'spearman'])
              trs.append(train_losses[-1])
              train_aucs.append(train_auc)
              if shots > 0:
                   ref_info.columns = ['max','min','sum', 'mean','var', 'c']
                   ref_c.append(ref_info['c'][0])
              else:
                   ref_info.columns = ['max','min','sum', 'mean','var']

              ref_max.append(ref_info['max'][0])
              ref_min.append(ref_info['min'][0])
              ref_sum.append(ref_info['sum'][0])
              ref_mean.append(ref_info['mean'][0])
              ref_var.append(ref_info['var'][0])
              channel_std.append(np.mean(ref_std.iloc[:,0]))
              assert len(ref_var) == len(sevs)
              logs_df = pd.concat([pd.DataFrame(oas, columns=['OA>0']),  pd.DataFrame(mid, columns=['OA>1']),pd.DataFrame(mid_2, columns=['OA>2']), pd.DataFrame(sevs, columns=['OA>3']), pd.DataFrame(sps, columns=['spearman']),
              pd.DataFrame(trs, columns=['train_loss']), pd.DataFrame(train_aucs, columns=['train_auc']),
              pd.DataFrame(ref_max, columns=['ref_max']), pd.DataFrame(ref_min, columns=['ref_min']), pd.DataFrame(ref_sum, columns=['ref_sum']), pd.DataFrame(ref_mean, columns=['ref_mean']), pd.DataFrame(ref_var, columns=['ref_var']),
              pd.DataFrame(ref_c, columns=['ref_centre']) , pd.DataFrame(channel_std, columns=['channel_std']) ,
              ], axis =1)
              if shots > 0:
                  logs_df=pd.concat([logs_df, pd.DataFrame(ref_c, columns=['ref_centre']) ], axis =1)

              if args.get_oarsi_results:
                  write_results(df, results, model_name, logs_df, current_epoch+epoch, model, optimizer, ref_std, args, oarsi_res)
              else:
                  write_results(df, results, model_name, logs_df, current_epoch+epoch, model, optimizer, ref_std, args)


              if test_dataset is not None:
                       if args.get_oarsi_results:
                           df, results, ref_info,ref_std,oarsi_res = evaluate_severity(patches, args.padding,args.patchsize, args.stride,seed, train_dataset, test_dataset, model, args.data_path, criterion, args.device, shots, args.meta_data_dir, args.get_oarsi_results)
                       else:
                           df, results, ref_info,ref_std = evaluate_severity(patches, args.padding,args.patchsize, args.stride,seed, train_dataset, test_dataset, model, args.data_path, criterion, args.device, shots, args.meta_data_dir, args.get_oarsi_results)


                       oas_test.append(results.loc[metric, 'auc'])
                       mid_test.append(results.loc[metric,'auc_mid'])
                       mid_2_test.append(results.loc[metric,'auc_mid2'])
                       sevs_test.append(results.loc[metric, 'auc_sev'])
                       sps_test.append(results.loc[metric, 'spearman'])


                       logs_df = pd.concat([pd.DataFrame(oas_test, columns=['OA>0']),  pd.DataFrame(mid_test, columns=['OA>1']),pd.DataFrame(mid_2_test, columns=['OA>2']), pd.DataFrame(sevs_test, columns=['OA>3']), pd.DataFrame(sps_test, columns=['spearman']),
                       pd.DataFrame(trs, columns=['train_loss']), pd.DataFrame(train_aucs, columns=['train_auc']),
                       pd.DataFrame(ref_max, columns=['ref_max']), pd.DataFrame(ref_min, columns=['ref_min']), pd.DataFrame(ref_sum, columns=['ref_sum']), pd.DataFrame(ref_mean, columns=['ref_mean']), pd.DataFrame(ref_var, columns=['ref_var']),
                      pd.DataFrame(channel_std, columns=['channel_std']) ,
                       ], axis =1)

                       if shots > 0:
                           logs_df=pd.concat([logs_df, pd.DataFrame(ref_c, columns=['ref_centre']) ], axis =1)

                       if args.get_oarsi_results:
                           write_results(df, results, model_name + '_on_test_set', logs_df, current_epoch+epoch, model, optimizer, ref_std, args, oarsi_res)
                       else:
                           write_results(df, results, model_name + '_on_test_set', logs_df, current_epoch+epoch, model, optimizer, ref_std, args)


      if (epoch % 10 == 0) & (eval_epoch ==0):
          if shots > 0:
              ref_c.append( centre_sim( args.padding,args.patchsize, args.stride, train_dataset, model, args.device, shots))
          eps.append(epoch)
          oas.append('na')
          mid.append('na')
          mid_2.append('na')
          sevs.append('na')
          sps.append('na')
          trs.append(train_losses[-1])
          train_aucs.append(train_auc)
          ref_max.append('na')
          ref_min.append('na')
          ref_sum.append('na')
          ref_mean.append('na')
          ref_var.append('na')
          channel_std.append('na')
          assert len(ref_var) == len(sevs)


          logs_df = pd.concat([pd.DataFrame(oas, columns=['OA>0']),  pd.DataFrame(mid, columns=['OA>1']),pd.DataFrame(mid_2, columns=['OA>2']), pd.DataFrame(sevs, columns=['OA>3']), pd.DataFrame(sps, columns=['spearman']),
            pd.DataFrame(trs, columns=['train_loss']), pd.DataFrame(train_aucs, columns=['train_auc']),
            pd.DataFrame(ref_max, columns=['ref_max']), pd.DataFrame(ref_min, columns=['ref_min']), pd.DataFrame(ref_sum, columns=['ref_sum']), pd.DataFrame(ref_mean, columns=['ref_mean']), pd.DataFrame(ref_var, columns=['ref_var']),
            pd.DataFrame(ref_c, columns=['ref_centre']) , pd.DataFrame(channel_std, columns=['channel_std']) ,  pd.DataFrame(eps, columns=['epoch'])
            ], axis =1)

          if shots > 0:
              logs_df=pd.concat([logs_df, pd.DataFrame(ref_c, columns=['ref_centre']) ], axis =1)


          logs_df.to_csv(os.path.join(args.dir_path, 'outputs/logs/')+ model_name)



      if (epoch == epochs):
            if args.get_oarsi_results:
                df, results,ref_info,ref_std,oarsi_res  = evaluate_severity(patches, args.padding,args.patchsize, args.stride,seed, train_dataset, val_dataset, model,  args.data_path, criterion, args.device, shots, args.meta_data_dir, args.get_oarsi_results)
                write_results(df, results, model_name, logs_df, current_epoch+epoch, model, optimizer, ref_std , args, oarsi_res)
            else:
                df, results,ref_info,ref_std  = evaluate_severity(patches, args.padding,args.patchsize, args.stride,seed, train_dataset, val_dataset, model,  args.data_path, criterion, args.device, shots, args.meta_data_dir, args.get_oarsi_results)
                write_results(df, results, model_name, logs_df, current_epoch+epoch, model, optimizer, ref_std , args)


            if test_dataset is not None:
                if args.get_oarsi_results:
                    df, results,ref_info,ref_std,oarsi_res = evaluate_severity(patches, args.padding,args.patchsize, args.stride,seed, train_dataset, test_dataset, model, args.data_path, criterion, args.device, shots, args.meta_data_dir, args.get_oarsi_results)
                    write_results(df, results, model_name + '_on_test_set', logs_df, current_epoch+epoch, model, optimizer, ref_std , args, oarsi_res)
                else:
                    df, results,ref_info,ref_std = evaluate_severity(patches, args.padding,args.patchsize, args.stride,seed, train_dataset, test_dataset, model, args.data_path, criterion, args.device, shots, args.meta_data_dir, args.get_oarsi_results)
                    write_results(df, results, model_name + '_on_test_set', logs_df, current_epoch+epoch, model, optimizer, ref_std, args)


  print("Finished Training")
   
