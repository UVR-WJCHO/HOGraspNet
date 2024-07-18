import os
import requests
import zipfile
from tqdm import tqdm
import time
import shutil

def download_file(url, local_path, retries=5):
    attempt = 0
    while attempt < retries:
        try:
            total_size = int(requests.head(url).headers['Content-Length'])
            if os.path.exists(local_path):
                temp_size = os.path.getsize(local_path)
                if total_size == temp_size:
                    print(f"File already fully downloaded: {local_path}")
                    return True
            else:
                temp_size = 0

            headers = {'Range': f'bytes={temp_size}-'}
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()  # Check if the request was successful

            with open(local_path, 'ab') as file, tqdm(
                desc=local_path,
                total=total_size,
                initial=temp_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                shutil.copyfileobj(response.raw, file, length=16*1024*1024)
                bar.update(total_size - temp_size)
                file.flush()
                os.fsync(file.fileno())

            if os.path.getsize(local_path) == total_size:
                print(f"Downloaded {local_path}")
                return True
            else:
                print(f"File size mismatch for {local_path}. Retrying...")
                attempt += 1
                time.sleep(2)  # Wait for 2 seconds before retrying
        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
            print(f"Error downloading {local_path}: {e}. Retrying ({attempt + 1}/{retries})...")
            attempt += 1
            time.sleep(2)  # Wait for 2 seconds before retrying
    print(f"Failed to download {local_path} after {retries} attempts.")
    return False

def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")
    except zipfile.BadZipFile:
        print(f"Failed to extract {zip_path}: Bad zip file.")

def read_urls_from_file(file_path):
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    return urls

def read_urls_from_directory(directory):
   
    return urls

# urls_zip_url = "http://data.uvrlab.org/datasets/HOGraspNet/urls.zip"

local_dir = "data"
# os.makedirs(local_dir, exist_ok=True)

# urls_zip_path = os.path.join(local_dir, "urls.zip")
# download_file(urls_zip_url, urls_zip_path)
# extract_zip(urls_zip_path, local_dir)

urls = []
basedir = os.path.join(os.getcwd(), 'assets/urls')
for filename in os.listdir("/scratch/NIA/HOGraspNet/assets/urls"):
    if filename.endswith("images.txt"):
        file_path = os.path.join("/scratch/NIA/HOGraspNet/assets/urls", filename)
        res = read_urls_from_file(file_path)
        urls.extend(res)

for url in urls:
    local_file_name = url.split('/')[-1]
    local_path = os.path.join(local_dir, local_file_name)
    download_file(url, local_path)
    break