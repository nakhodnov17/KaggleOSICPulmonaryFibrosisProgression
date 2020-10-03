import importlib
import time
from collections import namedtuple

import torch.utils.data

import torchio.transforms

from utils import CTDataset


if __name__ == '__main__':
    transforms = torchio.transforms.Compose([
        torchio.transforms.RandomAffine(
            degrees=(10, 10),
            translation=(-10, -10),
            isotropic=False,
            default_pad_value='minimum',
            image_interpolation='linear',
        ),
    ])

    dataset_train = CTDataset(
        './processed_data/train',
        './processed_data/train.csv',
        train=True, transform=transforms,
        test_size=0, random_state=42,
        padding_mode=None, padding_constant=None, pad_global=False
    )

    dl = torch.utils.data.DataLoader(dataset_train, batch_size=1, num_workers=3)

    st_time = time.time()
    for obj in dl:
        print(time.time() - st_time)
        print(obj[0])
        st_time = time.time()
