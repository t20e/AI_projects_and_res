

# ‼️‼️‼️ Bash script not needed.
#  It is a headache to use BASH or ZSH to create a complex setup pipeline that is portables across OSs. Instead I ended up using Python.
# Note: using python is easier to implement however you can not create a conda env with it, for example when the repo is downloaded and u call setup.py and lets say its job is to create the env and activate it but this will cause an error if setup.py has python modules that are not installed, we first what to create the env than activate it and this way when we use python the modules are already installed.

#!/bin/bash
set -euo pipefail # Exit program on error anywhere.
# setup.sh
# This script creates a virtual environment, installs dependencies, downloads dataset, and more.


# --- Global Variables & Configurations ---:
# dataset download link
readonly URL="https://www.kaggle.com/api/v1/datasets/download/gopalbhattrai/pascal-voc-2012-dataset"

readonly ROOT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
readonly DATASET_PATH="$ROOT_DIR/datasets"
readonly ZIP_FILE_PATH="$DATASET_PATH/pascal-voc-2012-dataset.zip"

# Environment 
readonly ENV_NAME="yolov1_env"
readonly YAML_FILE="./configs/environment.yaml"


# Import utils
source "$ROOT_DIR/utils/bash_script_utils.sh"

mkdir -p "$DATASET_PATH" # Create dataset directory if it doesn't exist


print_header "Setting Up YOLO Project"
sleep 2 # sleep for 2 seconds to allow user to see print
check_dependencies # Call function to check if command dependencies are installed.

# === CREATE ENVIRONMENT & INSTALL DEPENDENCIES ===
print_header "Creating Environment"

# check if Conda is installed
if ! command -v conda ; then
    error_exit "❌ Conda is not installed. Please install miniconda or Anaconda first."
fi

# check if the env already exists
if conda env list | grep -E "^$ENV_NAME[[:space:]]" > /dev/null 2>&1; then
    printf "\n✅ Conda env: $ENV_NAME is already created"
else
    printf "\n\n🚧 Creating Conda Environment from $YAML_FILE\n"
    conda env create -f "$YAML_FILE" || {
        error_exit "❌ Failed to create environment from $YAML_FILE."
    }
    echo "Created environment"
fi


# === DOWNLOAD DATASET ===
print_header "Downloading Dataset"

# Download the file
if [ ! -f "$ZIP_FILE_PATH" ] && [ ! -d "$DATASET_PATH/VOC2012_train" ] && [ ! -d "$DATASET_PATH/VOC2012_train_val" ] && [ ! -d "$DATASET_PATH/VOC2012_val" ]; then # check if the zip file is already downloaded
    printf "\n\n⬇️  Downloading Dataset: file is large, please wait.\n"
    curl -L --progress-bar -o "$ZIP_FILE_PATH" "$URL" || {
        rm -f "$FULL_FILE_PATH" # remove partially downloaded file on failure
        error_exit "❌ Failed to download file."
    }
    printf "\n✅ Zip file downloaded successfully.\n"
else
    # If the file already exists
    printf "\nDataset zip file already downloaded: $ZIP_FILE_PATH\n"
    echo -e "\n* If any issues occur and you want to re-download the dataset zip file, delete all the datasets belonging to the VOC dataset i.e VOC2012_train_val. Note: this doesn't download the TACO dataset."
fi

print_header "Un-zipping File"


if [ -f "$ZIP_FILE_PATH" ]; then # check if the zip file exists
    printf "\nUnzipping file: $ZIP_FILE_PATH\n\n"
    unzip_and_monitor "$ZIP_FILE_PATH" "$DATASET_PATH"
    # delete zip file
    rm "$ZIP_FILE_PATH"  
else
    printf "\nDataset Zip file doesn't exist at: $ZIP_FILE_PATH\n\n"
fi


# === SETUP FOLDER STRUCTURE ===
print_header "Structuring Project"

# --- Remove unnecessary nested folders
# i.e datasets/VOC2012_train_val/VOC2012_train_val -> to -> just datasets/VOC2012_train_val
if [ -d "$DATASET_PATH/VOC2012_train_val/VOC2012_train_val" ]; then # if dir exists
    remove_nested_folder "$DATASET_PATH"
fi

# --- Delete dataset directories and files that are not needed for this project.
if [ -d "$DATASET_PATH/VOC2012_train_val/ImageSets" ]; then # if dir exists
    rm -r "$DATASET_PATH/VOC2012_train_val/ImageSets"
    rm -r "$DATASET_PATH/VOC2012_train_val/SegmentationClass"
    rm -r "$DATASET_PATH/VOC2012_train_val/SegmentationObject"
fi

# test set
if [ -d "$DATASET_PATH/VOC2012_test/ImageSets" ]; then 
    rm -r "$DATASET_PATH/VOC2012_test/ImageSets"
fi

# --- Split the train and validation sets from the VOC dataset. (80/20) split.
if [ -d "$DATASET_PATH/VOC2012_train_val" ]; then
    split_train_val_sets "$DATASET_PATH"
fi

print_header "Project Setup Complete!"
# Tell user to activate environment
printf "\n⚠️ IMPORTANT ⚠️ \nActivate the environment! \n\n$             conda activate $ENV_NAME\n"