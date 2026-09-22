"""
Downloads the NSL-KDD dataset (KDDTrain+.txt and KDDTest+.txt) from a
public GitHub mirror and saves them under data/raw/.

Run directly:
    python -m ml.data.download_dataset
"""

import os
import urllib.request

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

FILES = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
}


def download_all(force: bool = False) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    for filename, url in FILES.items():
        dest = os.path.join(RAW_DIR, filename)
        if os.path.exists(dest) and not force:
            print(f"[skip] {filename} already exists at {dest}")
            continue
        print(f"[download] {filename} <- {url}")
        urllib.request.urlretrieve(url, dest)
        size_kb = os.path.getsize(dest) / 1024
        print(f"[done] {filename} saved ({size_kb:.1f} KB)")


if __name__ == "__main__":
    download_all()
