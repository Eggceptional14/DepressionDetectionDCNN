from torch.utils.data import Dataset
import pandas as pd


class DAICDataset(Dataset):
    def __init__(self, landmark_file, au_file, gaze_file, label_file=None, is_train=True):
        self.landmarks = pd.read_csv(landmark_file)
        self.aus = pd.read_csv(au_file)
        self.gaze = pd.read_csv(gaze_file)

        self.labels = None
        if label_file:
            self.labels = pd.read_csv(label_file)
        self.is_train = is_train

    def __len__(self):
        return (len(self.landmarks), len(self.aus), len(self.gaze))

    def __getitem__(self):
        pass