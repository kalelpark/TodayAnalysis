import argparse
import os
import shutil
import time
import numpy as np
from  utils import seed_everything, read_csv, save_model
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from pretrain import get_model
import datasetloader
from test import testing

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'            # SELECT left GPU 

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser( description = '[DACON] Super Resolution')

parser.add_argument('-TrainedModel', type = str,
                    help = 'Ex > -TrainedModel [TrainedModel] | Type : String')

parser.add_argument('-ImgSize', type = int,
                    help = 'Ex > -ImgSize [ImgSize] | Type : int')

parser.add_argument('-BatchSize', type = int,
                    help = 'Ex > -BatchSize [BatchSize] | Type : int')

parser.add_argument('-Epoch', type = int,
                    help = 'Ex > -Epoch [num] | Type : int')        

# parser.add_argument('-')

class config:
    global args
    args = parser.parse_args()

    seed = 42

    # ------ Data path ------
    train_path = "data/train.csv"
    test_path = "data/test.csv"
   
    modelname = args.TrainedModel
    imgsize = args.ImgSize
    batchsize = args.BatchSize
    epoch = args.Epoch
    lr = 1e-3

    fold = False
    transform = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

def training(model, dl_train):
    best_loss = 99999

    optimizer = torch.optim.Adam(params = model.parameters(), lr = config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # model.to(config.device)
    criterion = nn.MSELoss().to(config.device)

    for epochs in range(config.epoch):
        model.train()
        train_loss = []
        for lr_img, hr_img in dl_train:
            lr_img, hr_img = lr_img.float().to(config.device), hr_img.float().to(config.device)

            optimizer.zero_grad()

            pred_hr_img = model(lr_img)
            loss = criterion(pred_hr_img, hr_img)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        
        if scheduler is not None:
           scheduler.step()
        
        _train_loss = np.mean(train_loss)
        print(f'Epoch : [{epochs}] Train Loss : [{_train_loss:.5f}]')
         
        if best_loss > _train_loss:
            best_loss = _train_loss
            save_model(model, config.modelname)

def train(config):
    model = get_model(config.modelname)             # model test
    model = nn.DataParallel(model)                  # model test
    print(model.cuda())                             # model test

    dl_train, dl_test = datasetloader.get_dataloader()   

    # training(model, dl_train)                     # model test
    testing(config, dl_test)

if __name__ == '__main__':
    seed_everything(config.seed)
    print('============ START ============')
    train(config)
    