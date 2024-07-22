import os
import io
import torch
from torch.utils.data import Dataset
import pandas as pd
from dotenv import load_dotenv

from utils.downloader import file_processor

load_dotenv()


class DAICDataset(Dataset):

    def __init__(self, split_details, is_train=True):
        self.split_details = pd.read_csv(split_details) # read split of DAIC dataset (contains pid and labels)
        self.data_url = os.getenv('DAIC_DATA_URL')
        self.is_train = is_train

    def __len__(self):
        return len(self.split_details)

    def __getitem__(self, idx):
        pid = self.split_details['Participant_ID'].iloc[idx]

        file_url = f"{str(pid)}_P.zip"
        data = file_processor(self.data_url + file_url)
        print(f'##### Finish Downloading File: {file_url} #####\n')

        landmarks = pd.read_csv(io.BytesIO(data.get(f'{pid}_CLNF_features.txt')))
        aus = pd.read_csv(io.BytesIO(data.get(f'{pid}_CLNF_AUs.txt')))
        gaze = pd.read_csv(io.BytesIO(data.get(f'{pid}_CLNF_gaze.txt')))

        landmarks = landmarks[landmarks[' success'] == 1].iloc[:, 4:]
        aus = aus[aus[' success'] == 1].iloc[:, 4:]
        gaze = gaze[gaze[' success'] == 1].iloc[:, 4:]
        
        sample = {
            'pid': pid,
            'landmarks': torch.tensor(landmarks, dtype=torch.float32),
            'aus': torch.tensor(aus, dtype=torch.float32),
            'gaze': torch.tensor(gaze, dtype=torch.float32)
        }

        if self.is_train:
            label = self.split_details[self.split_details['Participant_ID'] == pid]['PHQ8_Binary'].values[0]
            sample["label"] = label

        return sample