# Starter code for Homework 3 of Representation Learning course @ USC.
# The course is delivered to you by the Information Sciences Institute (ISI).

import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import os
import sklearn
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics
import pandas as pd


class InOutMapping:

  def __init__(self):
      self.binary_vars = {}
      self.expand_colnames = []

  def map_input(self, features, dummies):
      expand_features = pd.get_dummies(features, columns=dummies)
      for c in dummies:
          if features[c].nunique() == 2:
              self.binary_vars[c] = features[c].unique()
              # deleting one of the binary feature columns
              new_col = c + '_' + self.binary_vars[c][1]
              expand_features = expand_features.drop(columns=[new_col])

      # features = features.drop(columns=['sex_Male', 'Y_<=50K'])
      print(expand_features.shape)
      self.expand_colnames = expand_features.columns
      scaler = sklearn.preprocessing.StandardScaler()
      X_scaled = scaler.fit_transform(expand_features)
      X = X_scaled.astype(np.float32)
      print(X.shape)
      return X

  def map_output(self, X, dummies, threshold=1):
      # TODO have the threshold
      onehot_cols = []
      noncat_cols_idx = []
      noncat_cols = []
      for idx, c in enumerate(self.expand_colnames):
          if any(map(c.startswith, dummies)):
              onehot_cols.append(c) #TODO maybe I don't need this
          else:
              noncat_cols_idx.append(idx)
              noncat_cols.append(c)

      revert_df = pd.DataFrame(columns=dummies)
      for c in dummies:
          dummy_idx = [idx for idx, val in enumerate(self.expand_colnames) if val.startswith(c)]
          print("**************************************")
          print(X.shape)
          corresponding = X[:, dummy_idx]
          print(dummy_idx)
          print(corresponding)
          if c in self.binary_vars.keys():
              b0 = self.binary_vars[c][0]
              b1 = self.binary_vars[c][1]
              values = [b1 if val < 0.5 else b0 for val in corresponding]  # TODO we don't have 0 and 1
          else:
              m = np.zeros_like(corresponding)
              m[np.arange(len(corresponding)), corresponding.argmax(1)] = 1
              max_col = np.argmax(m, axis=1)
              values = [self.expand_colnames[i].replace(c + '_', '') for i in max_col]
          revert_df[c] = values

      res = pd.DataFrame(X[:, noncat_cols_idx], columns=noncat_cols)
      for c in dummies:
          res[c] = revert_df[c]
      return res

os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)
np.random.seed(2019)
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
class VAEdSprite(nn.Module):
  
  def __init__(self, beta=4):
    super(VAEdSprite, self).__init__()
    self.lr = 5e-4
    self.beta = beta
    self.counter = 0
    self.C_max =Variable(torch.FloatTensor([20]))
    if torch.cuda.is_available():
          self.C_max=self.C_max.cuda()
    self.encoder = nn.Sequential(  
          # nn.Linear(58, 32), 
          # nn.ReLU(True),
          # nn.Linear(32, 16),         
          # nn.ReLU(True),
          # nn.Linear(16,8),   
          nn.Linear(104, 64), 
          nn.ReLU(True),
          nn.Linear(64, 32),         
          nn.ReLU(True),
          nn.Linear(32,8),         
      )

    self.decoder = nn.Sequential(
          # nn.Linear(4, 16),              
          # nn.ReLU(True),
          # nn.Linear(16, 32),                
          # nn.ReLU(True),
          # nn.Linear(32, 58), 
          nn.Linear(4, 32), 
          nn.ReLU(True),
          nn.Linear(32, 64),         
          nn.ReLU(True),
          nn.Linear(64,104),        
      )
    for block in self._modules:
        for m in self._modules[block]:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
              init.kaiming_normal(m.weight)
              if m.bias is not None:
                m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
              m.weight.data.fill_(1)
              if m.bias is not None:
                m.bias.data.fill_(0)

  def init_weights(self,m):
    if type(m) == nn.Linear:
      m.weight.data.fill_(1.0)
  def load(self):
    """Restores the model parameters. Called once by grader before sample_*."""
    # TODO(student): Implement.

  def pred(self, train_images):
    loaded_model = torch.load("model_dirgama10/best_modelgama10.pt", map_location=torch.device('cpu'))
    self.load_state_dict(loaded_model)
    test_data = torch.from_numpy(train_images).unsqueeze(1).float()
    enc = self.encoder(test_data)
    mean= enc[:,:4]
    covar = enc[:,4:]
    standard_div =(covar/2).exp()
    epsilon = Variable(standard_div.data.new(standard_div.size()).normal_())
    z = mean+epsilon*standard_div
    recon_x = self.decoder(z)
    recon_x = torch.nn.functional.sigmoid(recon_x)

    
  def pred2(self, female_train_images, male_train_images,dummies,df):
    #print(train_images)
    loaded_model = torch.load("model_dirgama10/best_modelgama10.pt", map_location=torch.device('cpu'))
    self.load_state_dict(loaded_model)
    for i in range(5):
      female_test_data = torch.from_numpy(female_train_images[i]).unsqueeze(0).float()
      male_test_data = torch.from_numpy(male_train_images[i]).unsqueeze(0).float()
      print(female_test_data.shape)
      female_enc = self.encoder(female_test_data)
      female_mean= female_enc[:,:4]
      female_covar = female_enc[:,4:]
      female_standard_div =(female_covar/2).exp()
      female_epsilon = Variable(female_standard_div.data.new(female_standard_div.size()).normal_())
      female_z = female_mean+female_epsilon*female_standard_div


      male_enc = self.encoder(male_test_data)
      male_mean= male_enc[:,:4]
      male_covar = male_enc[:,4:]
      male_standard_div =(male_covar/2).exp()
      male_epsilon = Variable(male_standard_div.data.new(male_standard_div.size()).normal_())
      male_z = male_mean+male_epsilon*male_standard_div



           
      inter = 0.4* female_z[0]+ (0.6)* male_z[0]
      print(inter.shape)
      recon_x = self.decoder(inter)
      recon_x = torch.nn.functional.sigmoid(recon_x)
      io_map = InOutMapping()
      io_map.map_input(df,dummies)
      res = io_map.map_output(recon_x.unsqueeze(0).detach().numpy(), dummies,1)
      print(res)



  def calculate_loss(self,x,recon_x,mean,covar):
    recon_loss = F.mse_loss(recon_x, x, size_average=False).div(64)
    kl_div_loss = 0.5*(-1 -covar+ (mean**2) +covar.exp())
    total_kl_loss =kl_div_loss.sum(1).mean(0, True)
    C = torch.clamp(self.C_max/1e5*self.counter, 0, self.C_max.data[0])
    total_loss = recon_loss + 100*(total_kl_loss-C).abs()
    return total_loss

  def fit(self, train_images):
    """Trains beta VAE.
    
    Args:
      train_images: numpy (uint8) array of shape [num images, 64, 64]. Only 2
        values are used: 0 and 1 (where 1 == "on" or "white"). To directly plot
        the image, you can visualize e.g. imshow(train_images[0] * 255).
    """
    #train_data = torch.from_numpy(train_images).unsqueeze(1).float()
    data_loader = DataLoader(train_images,batch_size=64,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
    best_loss = float("inf")
    br = True
    st = ''
    optimizer = optim.Adam(self.parameters(), lr=self.lr,betas=(0.9,0.999))
    while br:
      for x in data_loader:
        x=Variable(x)
        self.counter = self.counter+1
        # if counter == 5000:
        #   self.lr = self.lr/2
        # elif counter == 400000:
        #   self.lr = self.lr/2
        if torch.cuda.is_available():
          x=x.cuda()
        enc = self.encoder(x)
        mean= enc[:,:4]
        covar = enc[:,4:]
        standard_div = (covar/2).exp()
        epsilon = Variable(standard_div.data.new(standard_div.size()).normal_())
        z = mean+epsilon*standard_div
        recon_x = self.decoder(z)
        loss = self.calculate_loss(x,recon_x,mean,covar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l = loss.data[0]
        if loss < best_loss:
          best_loss=loss
          torch.save(self.state_dict(), 'model_dirgama10/best_modelgama10.pt')
          print("iter: " + str(self.counter) + " Loss "+ str(l) + "$")
        else:
          print("iter: " + str(self.counter) + " Loss "+ str(l))
        if self.counter >=15000:
          br = False
          break


    

if __name__ == '__main__':
  df_raw_male = pd.read_csv('male_adult_dataset', sep=', ', engine='python')
  df_raw_female = pd.read_csv('female_adult_dataset', sep=', ', engine='python')
  df_raw = pd.concat([df_raw_male, df_raw_female])
  #df_raw = df_raw.sample(frac=1).reset_index(drop=True)
  df = pd.get_dummies(df_raw, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'Y'])
  df = df.drop(columns=['sex_Male', 'Y_<=50K'])
  X = df.drop(columns=['Y_>50K'])
  scaler = sklearn.preprocessing.StandardScaler()
  X_scaled = scaler.fit_transform(df)
  X = X_scaled.astype(np.float32)
  #print(df.shape)
  #print(X_scaled.shape)
  vae = VAEdSprite()
  #vae.cuda()
  p  = df.values.astype(np.float32)
  #print(X)
  #vae.fit(X[1000:X.shape[0]-1000])
  print(X[0:1000].shape)
  vae.pred2(X[0:1000],X[-1000:],['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'Y'],df)
 