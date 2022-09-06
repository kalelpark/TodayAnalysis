import numpy as np
import pandas as pd
import os
import torch
import random

def seed_everything(seed):     # set seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_csv(train_path, test_path):    # read csv
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

def save_model(model, name):
    torch.save(model.state_dict(), f'savemodel/{name}_model.pt')
 