"""
Utilities for project setup
"""

import os
import zipfile
import requests
import os
import shutil
import random
from tqdm import tqdm


def print_header(statement: str):
    # Print Headers
    print("\n\n" + "-" * 64)
    print(statement)
    print("-" * 64)


def create_directory(directory_path: str):
    # Create a directory if it doesn't exist.
    # directory_path: The path to the directory to create.
    try:
        # Check if the directory already exists.
        if not os.path.exists(directory_path):
            # If it doesn't exist, create it.
            os.mkdir(directory_path)
            print(f"Directory -> '{directory_path}' created successfully.")
        else:
            # print(f"Directory -> '{directory_path}' already exists.")
            pass
    except OSError as e:
        print(f"Error creating directory '{directory_path}': {e}")


def download_file(url: str, save_path: str, filename: str):
    """
    Downloads a file from given URL.

    Args:
        url (str): The URL of the file to download.
        save_path: The directory where the file will be stored.
        filename (str): The name to save the file to.
    Returns:
        str: The zip files path.
    """

    if (
        not os.path.exists(f"{save_path}/{filename}")
        and not os.path.exists(f"{save_path}/VOC2012_train")
        and not os.path.exists(f"{save_path}/VOC2012_train_val")
        and not os.path.exists(f"{save_path}/VOC2012_test")
    ):
        try:
            print(f"Downloading VOC Dataset from '{url}'...")
            print("⬇️  file is large ~4GB, please wait.\n")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Add progress bar using tqdm
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 KB

            full_save_path = f"{save_path}/{filename}"

            with open(full_save_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

            print(f"✅ Downloaded to -> {full_save_path}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download file: {e}")
            return None
    else:
        print(
            f"Dataset zip '{filename}' already exists at -> '{save_path}/{filename}'. Skipping download."
        )
        print(
            "   If any issues occur and you want to re-download the dataset zip file, delete all the datasets belonging to the VOC dataset i.e VOC2012_train_val. Note: this doesn't download the TACO dataset."
        )
    return f"{save_path}/{filename}"


def unzip_file(save_path: str, zip_file_path: str):
    """
    Extracts a zip file.
    Args:
        save_path: The directory where the file will be stored.
        zip_file_path: Path to the zip file.
    """
    if (
        os.path.exists(zip_file_path)
        and not os.path.exists(f"{save_path}/VOC2012_test")
        and not os.path.exists(f"{save_path}/VOC_2012_dataset")
    ):
        print_header("Un-zipping File")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(save_path)


def structure_VOC(dataset_path):
    """Structure the VOC dataset directory.
    dataset_path (str): Path to the dataset directory.
    """

    voc_path = f"{dataset_path}/VOC_2012_dataset"
    # Create the mina VOC dataset directory
    if not os.path.exists(voc_path):
        os.mkdir(voc_path)
        # Create the validation directory
        create_directory(f"{voc_path}/train")
        create_directory(f"{voc_path}/test")

        # Move all the items from the train_val and test nested directories (/VOC2012_train_val/VOC2012_train_val etc..) to new voc_path/train etc..
        nested_dir = {
            "VOC2012_train_val": "train",  # Nested at -> VOC2012_train_val/VOC2012_train_val
            "VOC2012_test": "test",
        }
        for n in nested_dir:
            main_set_path = (
                f"{dataset_path}/{n}/{n}"  # i.e. datasets/VOC2012_test/VOC2012_test
            )
            for item in os.listdir(
                main_set_path
            ):  # item is one of either Annotations, JPEGImages, etc..
                item_path = f"{main_set_path}/{item}"
                destination_path = f"{voc_path}/{nested_dir[n]}"
                create_directory(destination_path)
                try:
                    shutil.move(item_path, destination_path)
                    # print(f"Moved '{item_path}' to '{destination_path}'")
                except shutil.Error as e:
                    # shutil.Error is raised for various issues, including permission errors
                    print(f"Error moving '{item_path}' to '{destination_path}': {e}")
                except OSError as e:
                    # General OS errors
                    print(f"An OS error occurred while moving '{item_path}': {e}")
            # If successful delete the nested directories that are no longer needed
            shutil.rmtree(f"{dataset_path}/{n}")  # rmtree() removes dir with content


def split_train_val(dataset_path):
    """
    Split the train_val sets into train and validation sets (80/20 split).

    Args:
        dataset_path (str): Path to the dataset directory
    """
    split_per = 0.20
    train_path = f"{dataset_path}/train"
    val_path = f"{dataset_path}/val"

    train_jpeg_path = os.path.join(train_path, "JPEGImages")
    train_annotations_path = os.path.join(train_path, "Annotations")

    val_jpeg_path = os.path.join(val_path, "JPEGImages")
    val_annotations_path = os.path.join(val_path, "Annotations")

    if not os.path.exists(val_path):
        print_header(
            "Spiting train_val set into train and validation sets (80/20 split)"
        )
        # Create destination directories if they don't exist
        create_directory(val_path)
        create_directory(val_jpeg_path)
        create_directory(val_annotations_path)

        # Get all image filenames (without extension) from the train JPEGImages folder
        image_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(train_jpeg_path)
            if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        # Calculate how many images to move
        num_images_to_move = int(len(image_files) * split_per)
        if (
            num_images_to_move == 0 and len(image_files) > 0
        ):  # Ensure at least one is moved if percentage is small
            num_images_to_move = 1

        # Randomly select image base names to move
        selected_image_bases = random.sample(image_files, num_images_to_move)

        moved_count = 0
        for (
            image_base
        ) in selected_image_bases:  # e.g. image_base = 2011_004208 without extension
            # Construct paths for the image and annotation pairs
            source_image_path = os.path.join(
                train_jpeg_path, f"{image_base}.jpg"
            )  # Assuming .jpg, adjust if other extensions
            source_annotation_path = os.path.join(
                train_annotations_path, f"{image_base}.xml"
            )

            dest_image_path = os.path.join(val_jpeg_path, f"{image_base}.jpg")
            dest_annotation_path = os.path.join(
                val_annotations_path, f"{image_base}.xml"
            )
            # Check if both source files (image and anno) exist before moving
            if os.path.exists(source_image_path) and os.path.exists(
                source_annotation_path
            ):
                try:
                    shutil.move(source_image_path, dest_image_path)
                    shutil.move(source_annotation_path, dest_annotation_path)
                    moved_count += 1
                    # print(f"Moved: {image_base}.jpg and {image_base}.xml")
                except Exception as e:
                    print(f"Error moving {image_base}: {e}")
            else:
                print(
                    f"Warning: Missing paired file for {image_base}. Skipping. (Image exists: {os.path.exists(source_image_path)}, Annotation exists: {os.path.exists(source_annotation_path)})"
                )

        print(f"\nSuccessfully moved {moved_count} image-annotation pairs.")
