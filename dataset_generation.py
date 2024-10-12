import pathlib

import mlstac

from utils import download_images, run_cloud_mask, select_hard_images

# Define the FOLDER where the dataset will be stored
DATASET_FOLDER = pathlib.Path("dataset")
DATASET_FOLDER.mkdir(exist_ok=True)

# Load the dataset
ds = mlstac.load(snippet="isp-uv-es/CloudSEN12Plus").metadata

# Subset the dataset
validation_subset = ds[
    (ds["split"] == "test") & (ds["label_type"] == "high") & (ds["proj_shape"] == 509)
]
validation_subset = validation_subset[validation_subset["clear_percentage"] < 50]
validation_subset.reset_index(drop=True, inplace=True)

# Download all the images (can take a bit of time)
download_images(dataset=validation_subset, data_folder=DATASET_FOLDER)

# Run the cloud masks
run_cloud_mask(folder=DATASET_FOLDER, device="cuda")

# Select the hardest images
dataframe_with_metrics = select_hard_images(folder=DATASET_FOLDER)
hard_images = dataframe_with_metrics[
    (
        dataframe_with_metrics["f2score_model_02"]
        + dataframe_with_metrics["f2score_model_01"]
        + dataframe_with_metrics["f2score_model_03"]
        < 2.90
    )
]
hard_images.reset_index(drop=True, inplace=True)
hard_images.to_csv(DATASET_FOLDER / "hard_images.csv", index=False)
