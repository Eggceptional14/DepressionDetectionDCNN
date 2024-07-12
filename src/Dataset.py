from torch.utils.data import Dataset
import pandas as pd


class DAICDataset(Dataset):
    def __init__(self, feature_file, label_file=None, is_train=True):
        self.visual_feature = pd.read_csv(feature_file)
        self.labels = None
        if label_file:
            self.labels = pd.read_csv()
        self.is_train = is_train

    def __len__(self):
        return len(self.visual_feature)

    def __getitem__(self):
        pass