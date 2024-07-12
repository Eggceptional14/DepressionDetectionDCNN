import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


def get_file_url(file_extension):
    data_url = os.getenv("DAIC_DATA_URL")
    response = requests.get(data_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    urls = []
    # Retrieve file url from webserver
    for link in soup.find_all('a'):
        file_url = link.get('href')
        if file_url.endswith(f'_P{file_extension}'):
            # print(file_url)
            urls.append(file_url)

    return urls

if __name__ == '__main__':
    print(get_file_url('.zip'))