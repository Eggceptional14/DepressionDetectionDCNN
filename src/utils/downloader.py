import requests
import io
import tarfile
import zipfile
import os
from dotenv import load_dotenv

load_dotenv()


def file_processor(file_url):
    # load tar file into memory
    response = requests.get(file_url)
    data = io.BytesIO(response.content)

    if file_url.endswith('.tar.gz'):
        extracted_data = tar_processor(data)
    elif file_url.endswith('.zip'):
        extracted_data = zip_processor(data)
    else:
        raise ValueError('Unsupported file format')

    return extracted_data

def tar_processor(data):
    # extract content of tar file
    with tarfile.open(fileobj=data, mode='r:gz') as tar:
        members = tar.getmembers()
        extracted_files = {
            member.name: tar.extractfile(member).read()
            for member in members
            if member.isfile()
        }
    return extracted_files

def zip_processor(data):
    # extract content of zip file
    with zipfile.ZipFile(data, 'r') as zip:
        extracted_files = {
            member: zip.read(member)
            for member in zip.namelist()
            if 'CLNF' in member and member.endswith('.txt')
        }

    return extracted_files

if __name__ == '__main__':
    print(file_processor(os.getenv("DAIC_TEST_URL")))