import os
import sys
import requests
import zipfile
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Not enough arguments")
        exit(1)
    
    sources = {
        "10": ["pss_10", "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"],
        "20": ["pss_20", "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip"]
    }

    key = sys.argv[1]
    name_source = sources.get(key)
    if name_source is None:
        print("[ERROR] Source for key not found")
        exit(1)

    name, source = name_source

    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / name
    zip_path = data_path / f"{name}.zip"

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)


    # Download pizza, steak, sushi data
    with open(zip_path, "wb") as f:
        request = requests.get(source)
        print(f"Downloading {name} data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print(f"Unzipping {name} data...") 
        zip_ref.extractall(image_path)

    # Remove zip file
    os.remove(zip_path)
else:
    current_file = Path(__file__).name
    print(f"{current_file} should be only used as entry point. Check transient dependencies")