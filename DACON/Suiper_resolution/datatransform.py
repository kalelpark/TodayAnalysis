import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0)],
        additional_targets={'image': 'image', 'label': 'image'}
    )

def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0)],
        additional_targets={'image': 'image', 'label': 'image'}
    )