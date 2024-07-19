import torch
from torch.utils.data import Dataset
import pandas as pd


class DAICDataset(Dataset):
    def __init__(self, pid, landmark_file, au_file, gaze_file, label_file=None, is_train=True):
        self.pid =pid
        self.landmarks = pd.read_csv(landmark_file)
        self.aus = pd.read_csv(au_file)
        self.gaze = pd.read_csv(gaze_file)

        self.labels = None
        if type(label_file) == str:
            self.labels = pd.read_csv(label_file)
        elif type(label_file) == pd.core.frame.DataFrame:
            self.labels = label_file
        self.is_train = is_train

    def __len__(self):
        return len(self.landmarks) # return frame length. All of the features have the same length

    def __getitem__(self, idx):
        landmarks = self.landmarks.iloc[idx, 4:]
        aus = self.aus.iloc[idx, 4:]
        gaze = self.gaze.iloc[idx, 4:]

        return (
            {"frame": idx, 
             "landmarks": landmarks, 
             "aus": aus, 
             "gaze": gaze, 
             "labels": self.labels[self.labels['Participant_ID'] == self.pid]}
            if self.is_train
            else {"frame": idx, "landmarks": landmarks, "aus": aus, "gaze": gaze}
        )
    
    # def __getitem__(self, idx):
    #     seq_id = self.sequences['PDB_ID'].iloc[idx]
    #     seq = self.sequences['SEQUENCE'].iloc[idx]
        
    #     if self.is_train:
    #         pssm_df = pd.read_csv(f"{self.pssm_dir}/{seq_id}_train.csv")
    #     else:
    #         pssm_df = pd.read_csv(f"{self.pssm_dir}/{seq_id}_test.csv")
    #     pssm_values = pssm_df.iloc[:, 2:].values
    #     pssm = torch.tensor(pssm_values, dtype=torch.float32)
    #     sample = {'seq': seq, 'pssm': pssm}

    #     if self.label_data is not None:
    #         labels = self.label_data[self.label_data['PDB_ID'] == seq_id]['SEC_STRUCT'].values[0]
    #         sample['labels'] = labels

    #     return sample