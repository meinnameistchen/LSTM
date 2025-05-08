# -*- coding: utf-8 -*-
import torch.utils.data as Data
from joblib import load

def dataloader(batch_size, workers=2):
    train_x = load('trainX_1024_10c')
    train_y = load('trainY_1024_10c')
    val_x = load('valX_1024_10c')
    val_y = load('valY_1024_10c')
    test_x = load('testX_1024_10c')
    test_y = load('testY_1024_10c')


    train_loader = Data.DataLoader(Data.TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = Data.DataLoader(Data.TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(Data.TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=True)
    return train_loader, val_loader, test_loader
