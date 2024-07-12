import torch
import os
import io
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from Model import MultiModalModel
from Dataset import DAICDataset
from downloader import file_processor
from url_extractor import get_file_url

load_dotenv()


def train():
    model = MultiModalModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    file_extension = '.zip'
    base_url = os.getenv('DAIC_DATA_URL')
    file_urls = get_file_url(file_extension)

    # loop through all files
    for file_url in file_urls:
        # load data into dataset class
        data = file_processor(base_url + file_url)
        landmarks = io.ByteIO(data.get(f'{file_url[:4]}CLNF_features.txt'))
        aus = io.ByteIO(data.get(f'{file_url[:4]}CLNF_AUs.txt'))
        gaze = io.ByteIO(data.get(f'{file_url[:4]}CLNF_gaze.txt'))
        dataset = DAICDataset(landmarks, 
                              aus, 
                              gaze, 
                              label_file='data/daic/train_split_Depression_AVEC2017.csv',
                              is_train=True)