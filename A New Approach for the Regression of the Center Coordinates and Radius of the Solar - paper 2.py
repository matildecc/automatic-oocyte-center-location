# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:43:11 2023

Based on: A New Approach for the Regression of the Center Coordinates and Radius of the Solar
Disk Using a Deep Convolutional Neural Network 
@author: Matilde
"""

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import cv2
from torchmetrics import R2Score

seed = 0
torch.manual_seed(seed)
#%% FUNCTIONS
#%% organizing data

class ImageDataset(Dataset):
  def __init__(self,df,img_folder,transform):
    self.df=df
    self.transform=transform
    self.img_folder=img_folder
    self.image_names = self.df[:]['image_name']
    self.labels = torch.Tensor(np.array([self.df[:]['center x'],self.df[:]['center y'] ]).T)
   
#The __len__ function returns the number of samples in our dataset.
  def __len__(self):
    return len(self.image_names)
 
  def __getitem__(self,index):
    image=Image.open(self.img_folder+str(self.image_names.iloc[index])).convert('RGB').resize((288,216))
    image=self.transform(image)
    targets=self.labels[index]
     
    return image, targets
#%%  Model
        
        
class VGGNet(nn.Module): 
    def __init__(self):
        super().__init__()
        #input 512x512x1
        self.feature_extractor = nn.Sequential(
            #B1
            nn.Conv2d(1, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            #B2
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            #B3
            nn.Conv2d(128, 256, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            #B4
            nn.Conv2d(256, 512, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 4, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding='same'),
            nn.ReLU(),

            )
        

        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(27*36*4, 256), 
            nn.ReLU(),
            nn. Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16,2), 
            nn.Identity()            
            )
        
        self.gradient = None
        
    def activations_hook(self, grad):
        self.gradient = grad

    def forward(self, x):
        #feature extractor
        x  = self.feature_extractor(x)
        h = x.register_hook(self.activations_hook)
        x = self.classifier(x)
        
        return x
    
    def get_activation_gradients(self):
        return self.gradient
    
    def get_activation(self, x):        
        return self.feature_extractor(x)
        
#%% Training functions 

#epoch method
def epoch_iter(dataloader, model, loss_fn, device, optimizer=None, is_train=True):
    global labels, preds, X, y, pred
    if is_train:
      assert optimizer is not None, "When training, please provide an optimizer."
      
    num_batches = len(dataloader)

    if is_train:
      model.train() # put model in train mode
    else:
      model.eval()

    total_loss = 0.0
    preds = []
    labels = []
    

    for batch, (X, y) in enumerate(tqdm(dataloader)):
        #y = y.type(torch.LongTensor) 
        X, y = X.to(device), y.to(device)
       

        # Compute prediction error
        pred = model(X)
        loss = torch.mean(torch.sqrt(torch.sum((pred - y)**2, axis = 1)))
        
        
                     
        #print(torch.sigmoid(pred))

        if is_train:
            
          #Regularization convolutions and weight kernels
          l2_lambda = 0.01
          l2_norm = 0
          for name, param in model.named_parameters():   
              if 'weight' in name: #only on the convolutions
                  l2_norm = param.pow(2.0).sum() + l2_norm
                  
          loss = loss + l2_lambda*l2_norm
          
          # Backpropagation
          optimizer.zero_grad() #sets the gradient to zero, before computed to not accumlate from the previous iter
          loss.backward() #The gradients are computed 
          optimizer.step() #performs a parameter update based on the current gradient
  
          
        # Save training metrics
        total_loss += loss.item() # IMPORTANT: call .item() to obtain the value of the loss WITHOUT the computational graph attached
        
       
    
        preds.extend(pred.cpu().detach().numpy())
        labels.extend(y.cpu().numpy())
        
        r2score = R2Score(num_outputs=2, multioutput='uniform_average')
        
    return total_loss / num_batches, r2score(torch.tensor(np.array(preds)), torch.tensor(np.array(labels)))
#%% aux functions

# auxiliary function to plot the loss and accuracy during training
def plotTrainingHistory(train_history, val_history, num_epochs):
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(range(1,num_epochs+1),train_history['loss'], label='train', color = 'black')
    plt.plot(range(1,num_epochs+1),val_history['loss'], label='val', color = 'red')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.title('Classification R2')
    plt.plot(range(1,num_epochs+1),train_history['R2'], label='train', color = 'black')
    plt.plot(range(1,num_epochs+1),val_history['R2'], label='val', color = 'red')
    
    


    plt.tight_layout()
    
    plt.show()
    
    

    
#%%
# main training method
def train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, num_epochs, model_name, device):
  train_history = {'loss': [], 'R2': []}
  val_history = {'loss': [], 'R2': [] }
  best_val_loss = np.inf
  val_loss_epoch_mult8 = np.inf
  train_loss = 0.0
  count = 0
  print("Start training...")
  
  for t in range(num_epochs):
         
    print(f"\nEpoch {t+1}")
    train_loss, train_r2 = epoch_iter(train_dataloader, model, loss_fn, device, optimizer)
    print(f"Train loss: {train_loss:.3f} \t Train R2: {train_r2:.3f} ")
    
    val_loss, val_r2 = epoch_iter(validation_dataloader, model, loss_fn,device, is_train=False)
    print(f"Val loss: {val_loss:.3f} \t Val R2: {val_r2:.3f}")

    #MUDAR - PQ N PODE SER ASSIM; TEM DE SER SEGUIDO - ler melhor 
    # save model when val loss improves
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
      torch.save(save_dict, model_name + '_best_val_loss.pth')     
    
       
    # save latest model
    save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t}
    torch.save(save_dict, model_name + '_latest_model.pth')
    
    if count == 0:
        val_loss_epoch_mult8 = val_loss
    
    if count%12 == 0 and count!= 0:
        if val_loss > val_loss_epoch_mult8:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']/10
                
            print( g['lr'])
        
            if (g['lr'] <= 2e-8):
                break
        
        val_loss_epoch_mult8 = val_loss
        
    count = count + 1  
    # save training history for plotting purposes
    train_history["loss"].append(train_loss)
    val_history["loss"].append(val_loss)
    
    train_history["R2"].append(train_r2)
    val_history["R2"].append(val_r2)
    

  print("Finished")
  return train_history, val_history, t
 

#%% RUNNING
#%% Data 
source_path =  "./database - immature/center cropped images/"
df = pd.read_excel('./database - immature/images_info.xlsx', dtype = str)

for path in os.listdir(source_path):   
    oocyte = (path.rpartition('-')[2][:-4])
    center_str = path.rpartition(' -')[0][9:]
    x = int(center_str.partition(',')[0])
    y = int(center_str.partition(',')[2])
    df.loc[df['oocyte n°'] == oocyte,'center x'] = x
    df.loc[df['oocyte n°'] == oocyte,'center y'] = y
    if not pd.isna((path)):
        df.loc[df['oocyte n°'] == oocyte,'image_name'] = path.rpartition('-')[2]

#new_df = df[(df["Accepted_AR (Y/N)"] == 'Y')]
new_df = df 

##Data Split
new_df_train = new_df.sample(n=round(0.80*len(new_df)), random_state= seed)
new_df_val = new_df[~new_df.isin(new_df_train)].dropna(subset = ['oocyte n°']).sample(n = round(0.10*len(new_df)), random_state= seed)
new_df_test = new_df[~new_df.isin(new_df_train) & ~new_df.isin(new_df_val)].dropna(subset = ['oocyte n°'])

img_folder_train =  "./database - immature/original images/"
transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale()
        ])


train_dataset = ImageDataset(new_df_train,img_folder_train,transforms_)
val_dataset = ImageDataset(new_df_val,img_folder_train,transforms_)
test_dataset =  ImageDataset(new_df_test,img_folder_train,transforms_)


#%% Augmentation  - flippings 


img_folder_trainH = "./database - immature/hflipped/"
img_folder_trainV = "./database - immature/vflipped/"
img_folder_trainVH = "./database - immature/vhflipped/"

new_df_train_H = new_df_train.copy()
new_df_train_V = new_df_train.copy()
new_df_train_VH = new_df_train.copy()

new_df_train_H.loc[:,'center x'] = (2592 - new_df_train['center x']) 
new_df_train_V.loc[:,'center y'] = 1944 -  new_df_train['center y']

new_df_train_VH.loc[:,'center x'] = 2592 - new_df_train['center x'] 
new_df_train_VH.loc[:,'center y'] = 1944 -  new_df_train['center y']  

train_datasetH =  ImageDataset(new_df_train_H,img_folder_trainH,transforms_)
train_datasetV = ImageDataset(new_df_train_V,img_folder_trainV,transforms_)
train_datasetVH = ImageDataset(new_df_train_VH,img_folder_trainVH,transforms_)


increased_dataset_train = torch.utils.data.ConcatDataset([train_dataset,
                                                          train_datasetH,
                                                          train_datasetV,
                                                          train_datasetVH,
                                                          ])
#%% Comment to Train

# if __name__ == '__main__':
     
#      # Load and Augmentation Data    
#      BATCH_SIZE = 16
#      NUM_WORKERS = 2
#      num_epochs = 150
        
#      device = "cuda" if torch.cuda.is_available() else "cpu"
#      print(f"Using {device} device")
     
     
#      train_dataloader = DataLoader(
#          increased_dataset_train, 
#          batch_size=BATCH_SIZE,
#          shuffle=True, 
#          num_workers=NUM_WORKERS, 
#          #pin_memory=True
         
#      )
      
#      validation_dataloader = DataLoader(
#          val_dataset, 
#          batch_size=BATCH_SIZE,
#          shuffle=False, 
#          #num_workers=NUM_WORKERS, 
#          #pin_memory=True
#      )
     
#      testing_dataloader = DataLoader(
#          test_dataset,
#          batch_size=BATCH_SIZE,
#          shuffle=False         
#          )
     
     
     
#      model = VGGNet()
#      model.to(device)
#      summary(model, (1,288, 216), batch_size = BATCH_SIZE)
     
     
#      def init_weights(m):
#         if isinstance(m, nn.Conv2d):
#             torch.nn.init.torch.nn.init.kaiming_uniform_(m.weight)
#             m.bias.data.fill_(0.0)

    
#      model.apply(init_weights)
     
#      #training
#      loss_fn = nn.MSELoss() # mudei para a distancia entre pontos
#      optimizer = optim.Adam(model.parameters(), lr=2e-4)
#      train_hist, val_hist, real_num_epochs = train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, num_epochs, model_name="OocyteCenter_morecomplex", device = device)
#      plotTrainingHistory(train_hist, val_hist,real_num_epochs)
     
     
#%% testing
#Saves the results on the folder center predictions - results, namely the euclidean distance between the true center and the predicted center 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

img_size = 256
BATCH_SIZE = 64
NUM_WORKERS = 2

model = VGGNet()
model.to(device)
#Load the best model to feedforward
model.load_state_dict(torch.load('models/OocyteCenter_morecomplex_best_val_loss.pth')['model'])
model.eval()

preds = []
labels = []
probabilities = []

indx = 0
  
for image, label in test_dataset:
    denorm_image = image.permute(1,2,0)
    image = image.unsqueeze(0).to(device)
    image_np = cv2.imread("./database - immature/original images/" + new_df_test.iloc[indx].image_name)
   
    # Compute prediction error
    pred_coords = model(image)
  
    # circle center
    cv2.circle(image_np, pred_coords.detach().cpu().numpy().astype(int)[0], 1, (255,255, 255), 3)
    # circle center
    cv2.circle(image_np, label.detach().cpu().numpy().astype(int), 1, (0, 0, 0), 3)
   
    
    cv2.imwrite("./center predictions - results/" + new_df_test.iloc[indx][0]+ 'ED-' + str(torch.sqrt(torch.sum((pred_coords.cpu() - label.cpu())**2, axis = 1)).cpu().detach().numpy()[0]) + '.png', image_np)
    
    
    preds.extend(pred_coords.cpu().detach().numpy())
    labels.append(label.cpu().detach().numpy())
    indx = indx + 1
plt.figure()
plt.scatter(np.array(labels)[:,0], np.array(preds)[:,0])
plt.plot(np.array(labels)[:,0], np.array(labels)[:,0])
plt.xlabel('True coordinate of the x')
plt.ylabel('Predicted coordinate of the x')
plt.title('R2 = ' + str(round(metrics.r2_score(np.array(labels)[:,0], np.round(np.array(preds)[:,0])),3)))

plt.figure()
plt.scatter(np.array(labels)[:,1], np.array(preds)[:,1])
plt.plot(np.array(labels)[:,1], np.array(labels)[:,1])
plt.xlabel('True coordinate of the y')
plt.ylabel('Predicted coordinate of the y')
plt.title('R2 = ' + str(round(metrics.r2_score(np.array(labels)[:,1], np.round(np.array(preds)[:,1])),3)))
   

         