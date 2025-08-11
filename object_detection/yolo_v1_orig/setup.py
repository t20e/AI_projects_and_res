"""
Setup Project

⭐️ Using Python to create and manipulate (directories, files, etc...) is more portable (between Max, Windows, Linux etc..) and simpler to implement than using shells like BASH or ZSH.
"""

import os
import time
import sys

from utils.setup_utils import *


if __name__ == "__main__":

    # Get all arguments passed to the script.
    args = sys.argv

    # --- Global Variables & Configurations ---
    # Dataset download link
    VOC_URL = "https://www.kaggle.com/api/v1/datasets/download/gopalbhattrai/pascal-voc-2012-dataset"
    # Test url -> VOC_URL = "https://github.com/t20e/res/raw/refs/heads/main/test-can-del/test.zip"

    # Dataset Paths
    CWD = os.getcwd()
    DATASET_PATH = os.path.join(CWD, "data/datasets")

    # Create directories that are not committed to repo.
    create_directory(DATASET_PATH)
    create_directory(f"{CWD}/model/saved_models")
    create_directory(f"{CWD}/model/checkpoints")
    create_directory(f"{CWD}/model/pre_trained")

    print_header("Setting Up YOLO Project...")
    time.sleep(2)  # Sleep for 2 seconds to allow user to read prints.

    if "--download_VOC" in args:
        # === DOWNLOAD THE VOC DATASET ===
        # Download the dataset
        print_header("Downloading Dataset")
        zip_file_path = download_file(VOC_URL, DATASET_PATH, filename="VOC_dataset.zip")
        zip_file_path = f"{DATASET_PATH}/VOC_dataset.zip"  # just in case

        # Extract zip file
        unzip_file(DATASET_PATH, zip_file_path)

        # === SETUP FOLDER STRUCTURE ===
        print_header("Structuring Project")

        structure_VOC(dataset_path=DATASET_PATH)

        split_train_val(f"{DATASET_PATH}/VOC_2012_dataset")

    print_header("Project setup Complete!")
