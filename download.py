import os
from PIL import Image
import sys
from concurrent.futures import ThreadPoolExecutor
import time
import requests
import hashlib
from tqdm import tqdm
import concurrent.futures


def download_image(url, output_folder, name=None):
    try:
        if name is None:
            name = os.path.basename(url)
        output_path = os.path.join(output_folder, name)
        os.system(f"wget -q -O {output_path} {url}")
    except Exception as e:
        print(f"Failed to download image {url}: {e}")


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("Usage: python downlaod.py <file name> <directory>")
        sys.exit(1)

    file_name = sys.argv[1]

    if not os.path.isfile(file_name):
        print(f"The file {file_name} does not exist.")
        sys.exit(1)

    directory = sys.argv[2]
    # create folder if not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # with open(file_name, "r") as f:
    #     urls = f.readlines()

    # start_time = time.time()

    # urls = list(set(urls))

    # total_urls = len(urls)
    # with ThreadPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(download_image, url.strip(), directory, f"{time.strftime('%Y%m%d%H%M')}_{i+1}.jpg") for i, url in enumerate(urls)
    #     ]
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=total_urls, desc="Downloading images", unit="image"):
    #         future.result()

    # end_time = time.time()
    # print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")

    def calculate_md5(file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def remove_zero_byte_files(directory):
        zero_byte_files = 0
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    zero_byte_files += 1
                    print(f"Removed zero-byte file: {file_path}")
        print("\nZero-byte file removal complete.")

    def remove_duplicates(directory):
        md5_dict = {}
        processed_files = 0
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_md5 = calculate_md5(file_path)
                if file_md5 in md5_dict:
                    os.remove(file_path)
                    print(f"Removed duplicate file: {file_path}")
                else:
                    md5_dict[file_md5] = file_path
                processed_files += 1
        print("\nDuplicate removal complete.")

    remove_zero_byte_files(directory)
    remove_duplicates(directory)