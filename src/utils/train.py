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


def train():
    model = MultiModalModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    base_url = os.getenv('EDAIC_DATA_URL')
    file_urls = get_file_url()

    for file_url in file_urls:
        data = file_processor(base_url + file_url)
        dataset = DAICDataset(io.ByteIO(data.get()))