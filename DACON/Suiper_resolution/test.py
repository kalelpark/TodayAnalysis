import os
import cv2
import numpy as np
import zipfile
import torch
import torch.nn as nn
from utils import seed_everything, read_csv, save_model
from pretrain import get_model
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
from pretrain import get_model
import datasetloader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
import PIL


def testing(config, dl_test):   # If you want to change test => train.py line 101

    model = get_model(config.modelname)
    model.load_state_dict(torch.load('savemodel/SRCNN_model.pt'), strict = False)
    model.eval()
    model.to(config.device)

    lr_img = cv2.imread('train/lr/0000.png')
    img_t = torchvision.transforms.ToTensor()(lr_img)
    # # np_arr = np.array(img_t)
    # img = plt.imshow()
    # img.save('path')
    pred_model = model(img_t.to(config.device))
    pred_model = pred_model.permute((1, 2, 0))
    pred_model = pred_model.to('cpu')  
    pred_model = pred_model.detach().numpy()
    print('222')
    plt.imshow(pred_model*(10**3))
    print(pred_model)
    plt.savefig('argdgfd')

    # pred_img_list = []
    # name_list = []

    # with torch.no_grad():
    #     for lr_img, file_name in dl_test:
    #         lr_img = lr_img.float().to(config.device)

    #         pred_hr_img = model(lr_img)

    #         for pred, name in zip(pred_hr_img, file_name):
    #             pred = pred.cpu().clone().detach().numpy()
    #             pred = pred.transpose(1, 2, 0)
    #             pred = pred * 255.

    #             pred_img_list.append(pred.astype('uint8'))
    #             name_list.append(name)

    # os.makedirs('./submission', exist_ok = True)
    # os.chdir("./submission/")

    # sub_imgs = []
    
    # for path, pred_img in zip(name_list, pred_img_list):
    #     cv2.imwrite(path, pred_img)
    #     sub_imgs.append(path)

    # submission = zipfile.ZipFile("../submission.zip", 'w')
    # for path in sub_imgs:
    #     submission.write(path)
    # submission.close()
    print('done')      

