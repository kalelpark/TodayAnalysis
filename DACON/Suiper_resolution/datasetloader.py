import cv2
from torch.utils.data import Dataset
from train import config
from torch.utils.data import DataLoader
from utils import read_csv
from datatransform import get_train_transform, get_test_transform

class CustomDataset(Dataset):
    def __init__(self, df, train_mode, transforms):
        self.df = df
        self.train_mode = train_mode
        self.transform = transforms
    
    def __getitem__(self, index):
        lr_path = self.df['LR'].iloc[index]
        lr_img = cv2.imread(lr_path)
        lr_img = cv2.resize(lr_img, (config.imgsize, config.imgsize), interpolation = cv2.INTER_CUBIC)

        if self.train_mode:
            hr_path = self.df['HR'].iloc[index]
            hr_img = cv2.imread(hr_path)
            
            if self.transform is not None:
                transformed = self.transform(image=lr_img, label=hr_img)
                lr_img = transformed['image'] / 255.
                hr_img = transformed['image'] / 255.
            return lr_img, hr_img

        else:
            file_name = lr_path.split('/')[-1]
            if self.transform is not None:
                print("Hello")
                transformed = self.transform(image = lr_img)
                lr_img = transformed['image'] / 255.
            return lr_img, file_name
        
    def __len__(self):
        return len(self.df)

def get_dataloader():
    train_df, test_df = read_csv(config.train_path, config.test_path)
    
    ds_train = CustomDataset(df = train_df, train_mode = True, transforms = get_train_transform())
    ds_test = CustomDataset(df = test_df, train_mode = False, transforms = None)

    dl_train  = DataLoader(ds_train, batch_size = config.batchsize, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size = config.batchsize, shuffle=False)

    return dl_train, dl_test


