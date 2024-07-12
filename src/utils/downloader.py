import requests
import io
import tarfile
import os
from dotenv import load_dotenv

load_dotenv()


def file_processor(file_url):
    # load tar file into memory
    response = requests.get(file_url)
    data = io.BytesIO(response.content)

    # extract content of tar file
    with tarfile.open(fileobj=data, mode='r:gz') as tar:
        members = tar.getmembers()
        extracted_data = {
            member.name: tar.extractfile(member).read()
            for member in members
            if member.isfile()
        }

    return extracted_data


if __name__ == '__main__':
    print(file_processor(os.getenv("TEST_URL")))