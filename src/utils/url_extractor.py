import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


def get_file_url():
    data_url = os.getenv("EDAIC_DATA_URL")
    response = requests.get(data_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    urls = []
    for link in soup.find_all('a'):
        file_url = link.get('href')
        if file_url.endswith('.tar.gz'):
            # print(file_url)
            urls.append(file_url)

    return urls