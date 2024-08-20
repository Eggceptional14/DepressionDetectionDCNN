import os
import io
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from utils.downloader import file_processor

load_dotenv()


class DAICDataset(Dataset):

    def __init__(self, split_details, chunk_size, is_train=True):
        self.split_details = pd.read_csv(split_details) # read split of DAIC dataset (contains pid and labels)
        self.data_url = os.getenv('DAIC_DATA_URL')
        self.is_train = is_train
        self.from_web = False
        # self.frame_step = step
        self.chunk_size = chunk_size

        self.chunk_info = self._calculate_chunks()

    def __len__(self):
        return sum(num_chunks for _, num_chunks in self.chunk_info)
    
    def _calculate_chunks(self):
        chunk_info = []
        base_url = os.getenv('DISK_DIR')
        # base_url = os.getenv('WIN_DIR')
        
        for idx, row in self.split_details.iterrows():
            pid = int(row['Participant_ID'])
            
            landmarks = pd.read_csv(f'{base_url}{pid}/{pid}_CLNF_features.csv', low_memory=False)
            aus = pd.read_csv(f'{base_url}{pid}/{pid}_CLNF_AUs.csv')
            gaze = pd.read_csv(f'{base_url}{pid}/{pid}_CLNF_gaze.csv')

            landmarks = landmarks[landmarks[' success'] == 1]
            aus = aus[aus[' success'] == 1]
            gaze = gaze[gaze[' success'] == 1]

            valid_indices = landmarks.index.intersection(aus.index).intersection(gaze.index)

            # calculate chunk info
            valid_frames = landmarks.loc[valid_indices].shape[0]
            num_chunks = (valid_frames + self.chunk_size - 1) // self.chunk_size
            chunk_info.append((pid, num_chunks))
        
        return chunk_info
    
    def _load_chunk_data(self, pid, chunk_idx):
        if self.from_web:
            file_url = f"{str(pid)}_P.zip"
            print(f'##### Start Downloading File: {file_url} #####\n')
            data = file_processor(self.data_url + file_url)
            print('##### Download Complete #####')

            landmarks = pd.read_csv(io.BytesIO(data.get(f'{pid}_CLNF_features.csv')), low_memory=False)
            aus = pd.read_csv(io.BytesIO(data.get(f'{pid}_CLNF_AUs.csv')))
            gaze = pd.read_csv(io.BytesIO(data.get(f'{pid}_CLNF_gaze.csv')))
        else:
            base_url = os.getenv('DISK_DIR')
            # base_url = os.getenv('WIN_DIR')
            
            landmarks = pd.read_csv(f'{base_url}{pid}/{pid}_CLNF_features.csv', low_memory=False)
            aus = pd.read_csv(f'{base_url}{pid}/{pid}_CLNF_AUs.csv')
            gaze = pd.read_csv(f'{base_url}{pid}/{pid}_CLNF_gaze.csv')
        
        # Align indices to ensure consistency across landmarks, AUs, and gaze
        valid_indices = landmarks.index.intersection(aus.index).intersection(gaze.index)

        # Select only the valid frames that exist in all features
        landmarks = landmarks.loc[valid_indices].iloc[:, 4:].values.astype(np.float32)
        aus = aus.loc[valid_indices].iloc[:, 4:].values.astype(np.float32)
        gaze = gaze.loc[valid_indices].iloc[:, 4:].values.astype(np.float32)

        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, len(landmarks))

        sample = {
            'pid': pid,
            'landmarks': torch.tensor(landmarks[start: end], dtype=torch.float32),
            'aus': torch.tensor(aus[start: end], dtype=torch.float32),
            'gaze': torch.tensor(gaze[start: end], dtype=torch.float32)
        }
        # print(sample)

        if self.is_train:
            label = self.split_details[self.split_details['Participant_ID'] == pid]['PHQ8_Binary'].values[0]
            sample["label"] = label
        else:
            actual = self.split_details[self.split_details['Participant_ID'] == pid]['PHQ_Binary'].values[0]
            sample["actual"] = actual

        return sample

    def __getitem__(self, idx):
        cumulative_chunks = 0
        for pid, num_chunks in self.chunk_info:
            if idx < cumulative_chunks + num_chunks:
                chunk_idx = idx - cumulative_chunks
                return self._load_chunk_data(pid, chunk_idx)
            cumulative_chunks += num_chunks

        raise IndexError("Index out of range")