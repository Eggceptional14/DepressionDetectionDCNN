import torch
import os
import io
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import torch.optim as optim
from torch.utils.data import DataLoader

from Model import CNNModel, LSTMModel, MultiModalModel
from Dataset import DAICDataset
from utils.downloader import file_processor
from utils.url_extractor import get_file_url

load_dotenv()


def train():
    config = {
        "dropout": tune.loguniform(0.2, 0.4),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64]),
    }
    model = MultiModalModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optim = optim.Adam(model.parameters())

    file_extension = '.zip'
    base_url = os.getenv('DAIC_DATA_URL')
    # file_urls = get_file_url(file_extension)
    train_labels = pd.read_csv("data/daic/train_split_Depression_AVEC2017.csv")
    print("##### Model Training #####\n\n")

    # loop through all files
    for pid in train_labels['Participant_ID']:
        # load data into dataset class
        file_url = f"{str(pid)}_P.zip"
        data = file_processor(base_url + file_url)
        print(f'##### Finish Downloading File: {file_url} #####\n\n')

        landmarks = io.BytesIO(data.get(f'{pid}_CLNF_features.txt'))
        aus = io.BytesIO(data.get(f'{pid}_CLNF_AUs.txt'))
        gaze = io.BytesIO(data.get(f'{pid}_CLNF_gaze.txt'))

        dataset = DAICDataset(pid,
                              landmarks, 
                              aus, 
                              gaze, 
                              label_file=train_labels,
                              is_train=True)

        print(f'Number of Frames in Dataset: {dataset.__len__()}')
        print(dataset.__getitem__(0))

        break
        
if __name__ == "__main__":
    train()