"""
Dataframe utils
"""

import os
import pandas as pd
import gc  # built-in memory clean-up
from PIL import Image


def create_df(dataset_path: str, num_to_load: int = 0, save_to_csv: bool = False):
    """
    Creates a dataframe of corresponding image and label file_names by row.
        Example CSV format (e.g., test.csv):
            | Image   | Label     |
            |---------|-----------|
            | 1.jpeg  | 124.xml   |
            | 2.jpeg  | 116.xml   |
            | ...     | ...       |
    Args:
        dataset_path (str): Path to a directory inside the dataset directory.
            - Path to the dataset. i.e. datasets/VOC2012_train_val
            - Needs to contain two directories -> /Annotations and /JPEGImages
        num_to_load (int): Number of images/annotations to load.
            - Set to 0 to load entire dataframe.
        save_to_csv (bool): Whether to save dataframe to csv.

    """
    imgs_dir = os.path.join(dataset_path, "JPEGImages")
    # annos for annotations
    annos_path = os.path.join(dataset_path, "Annotations")

    # --- Get the image and label filenames and store in a sorted list.
    image_files = sorted(
        [f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f))]
    )
    anno_files = sorted(
        [
            f
            for f in os.listdir(annos_path)
            if os.path.isfile(os.path.join(annos_path, f))
        ]
    )

    # --- If num_to_load > 0, only grab that much imgs/annotations.
    if num_to_load > 0:
        image_files = image_files[:num_to_load]
        anno_files = anno_files[:num_to_load]

    # --- Match files by name.
    matched_data = []
    # Note: I could just combine the two sorted arrays into a df but that could invite some issues like missing Pairs.
    for img_f in image_files:
        anno_xml = (
            img_f.rsplit(".", 1)[0] + ".xml"
        )  # Grab the image filename but remove the .jpg and add .xml, so we can compare vs the anno xml files.
        if (
            anno_xml in anno_files
        ):  # check if the the filename exists in the label directory.
            matched_data.append({"img": img_f, "annotation": anno_xml})

    # --- Create dataframe
    df = pd.DataFrame(matched_data)

    ## Delete the df and clean up memory.
    ## gc.collect()

    # --- Save to csv?
    if save_to_csv:
        # If custom size add size to filename.
        file_name = "image_label_df"
        if num_to_load > 0:
            file_name += f"_size_{num_to_load}"
        file_path = f"{dataset_path}/{file_name}.csv"
        # --- check if the file exists
        f"\Checking if dataframe csv file with {num_to_load} size already exists."
        if not os.path.exists(file_path):
            print(
                f"\n\nSaving dataset with {num_to_load} examples to CSV file ->", file_path
            )
            df.to_csv(file_path, index=False)

    return df


def test():
    cwd = os.getcwd()
    dataset_path = os.path.join(cwd, "datasets", "VOC2012_train_val")
    print(create_df(dataset_path, num_to_load=10, save_to_csv=False))

if __name__ == '__main__':
    test()
