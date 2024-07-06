import pandas as pd
import numpy as np
import os
from skimage import io
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SEED = 42


class CarDataSet(Dataset):
    """Car images dataset, each image has two labels for the car's hood and left backdoor"""

    def __init__(self, car_df: pd.DataFrame, images_dir: str, transform=None) -> None:
        """
        Args:
            car_df: the data frame with annotations.
            images_dir: directory with all the images.
            transform (callable, optional): transformations to apply on the images
        """
        self.car_df = car_df
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.car_df)

    def __getitem__(self, idx: int) -> tuple:
        labels = np.array([self.car_df.iloc[idx, -2:]], dtype=np.float32)
        labels = np.squeeze(labels)
        image_path = os.path.join(self.images_dir, self.car_df.iloc[idx, 0])
        image = io.imread(image_path)
        if self.transform:
            image = self.transform(image)
        return image, labels


def split_df_labels() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct and then then split the labels data frame into train, validation, and test data frames
    Returns:
        train_df: the training labels data frame
        val_df:  the validation labels data frame
        test_df: the test labels data frame
    """
    file_name = list(filter(lambda file: file.endswith(".csv"), os.listdir("labels")))[
        0
    ]
    df_labels = pd.read_csv(f"labels/{file_name}")
    train_val_df, test_df = train_test_split(
        df_labels, test_size=TEST_RATIO, random_state=SEED, shuffle=True
    )
    train_df = train_val_df.iloc[
        : int(len(train_val_df) * TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO))
    ]
    val_df = train_val_df.iloc[
        int(len(train_val_df) * TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO)) :
    ]
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    return train_df, val_df, test_df
