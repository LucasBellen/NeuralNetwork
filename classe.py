import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.nn import MSELoss


class Data(pl.LightningDataModule):
    
    def __init__(self, dataset: pd.DataFrame, labels_columns: str, drop_columns: str, bool_columns = None, batch_size=64):
        super().__init__()
        
        self.drop_columns = drop_columns
        self.bool_columns = bool_columns
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels_columns = labels_columns
    
    
    def setup(self, stage = None):
    
        if self.bool_columns:
            for names in self.bool_columns:
                if self.dataset[names].nunique() != 2:
                    raise ValueError(f'A coluna {names} não é binária.')
                else:
                    self.dataset[names] = np.where(self.dataset[names] == self.dataset[names][1], 1, 0)

        X = self.dataset.drop(columns= self.drop_columns)
        X = X.drop(columns= self.labels_columns).values
        X = np.expand_dims(X, axis=1)
        y = self.dataset[self.labels_columns].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.25)
        
    def train_dataloader(self):
        self.train_dataset = DataLoader(TensorDataset(torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32)), batch_size=self.batch_size)
        return self.train_dataset
    
    def val_dataloader(self):
        self.val_dataset = DataLoader(TensorDataset(torch.tensor(self.X_val, dtype=torch.float32), torch.tensor(self.y_val, dtype=torch.float32)), batch_size=self.batch_size)
        return self.val_dataset
    
    def test_dataloader(self):
        self.test_dataset = DataLoader(TensorDataset(torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.float32)), batch_size=self.batch_size)   
        return self.test_dataset
   

class rede(pl.LightningModule):

    def __init__(self):
        super().__init__()
        
        self.convolutionals = nn.Sequential(nn.Conv1d(in_channels=1, out_channels= 32, kernel_size= 4),
                                            nn.ELU(),
                                            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                            nn.Conv1d(in_channels=32, out_channels= 64, kernel_size= 6),
                                            nn.ELU(),
                                            nn.MaxPool1d(kernel_size=4, stride=4, padding=2))
        
        self.linear = nn.Linear(64, 64)
        self.flat = nn.Flatten()
        self.out = nn.Linear(64,1)
        self.loss_fn = MSELoss()
        
    def forward(self, x : Tensor) -> Tensor:
        
        x = self.convolutionals(x)
        x = self.flat(x)
        x = self.linear(x)
        x = F.relu(x)
        x = x.view(x.size(0),-1)
        x = self.out(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y.unsqueeze(1))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y.unsqueeze(1))
        return loss       
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y.unsqueeze(1))
        return loss       
    
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3)    












# aprimorar com o pacote tabular

# https://github.com/Alcoholrithm/TabularS3L
# https://medium.com/@david.tiefenthaler/pytoch-lightning-tabular-classification-312d6b753d28
