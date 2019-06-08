import io
from zipfile import ZipFile

import requests


def download_extract_zip(url, dir="./EgoHands_dataset/"):
    response = requests.get(url)
    with ZipFile(io.BytesIO(response.content)) as thezip:
        thezip.extractall(dir)


url = "http://vision.soic.indiana.edu/egohands_files/egohands_videos.zip"
