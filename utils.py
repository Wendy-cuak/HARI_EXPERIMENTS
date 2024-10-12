import pathlib
import pickle

import matplotlib.pyplot as plt
import mlstac
import numpy as np
import pandas as pd
import rasterio as rio
import requests
import segmentation_models_pytorch as smp
import torch
from phicloudmask import CloudMask
from phicloudmask.constant import SENTINEL2_DESCRIPTORS
from rasterio import CRS, Affine
from sklearn.metrics import fbeta_score


def download_images(dataset, data_folder):
    (data_folder / "raw").mkdir(parents=True, exist_ok=True)
    for idx, row in dataset.iterrows():
        print(f"Processing {idx + 1}/{len(dataset)}")
        datapoint, metadata = mlstac.get_data(
            dataset=dataset.iloc[idx], save_metadata_datapoint=True
        )[0]
        metadata["quality"] = 100
        metadata["reversible"] = True
        metadata["ycbcr420"] = True
        name = row["datapoint_id"]
        with rio.open(data_folder / f"raw/{name}.tif", "w", **metadata) as src:
            src.write(datapoint)


def cloud_model2(outpath="model", device="cpu"):
    # Download the model
    model_link = "https://huggingface.co/datasets/isp-uv-es/CloudSEN12Plus/resolve/main/demo/models/UNetMobV2_V2.pt"
    model_path = pathlib.Path(outpath) / "UNetMobV2_V2.pt"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(model_link, stream=True) as r:
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

    # Load the weights into the model
    model_v2 = smp.Unet(
        encoder_name="mobilenet_v2", encoder_weights=None, classes=4, in_channels=13
    )
    model_v2.load_state_dict(torch.load(model_path))

    # Desactivate the gradient estimation
    for param in model_v2.parameters():
        param.requires_grad = False

    # To eval model
    model_v2 = model_v2.eval()
    model_v2 = model_v2.to(device)

    return model_v2


def cloud_model3(outpath="model", device="cpu"):
    # Define the semantic categories mapping
    cloudsen12_style = {
        0: 0,
        1: 0,
        2: 0,
        6: 0,  # Merged into category 0 (land, water, snow, no_data)
        4: 1,  # Thick cloud -> category 1
        3: 2,  # Thin cloud -> category 2
        5: 3,  # Shadow -> category 3
    }
    map_values = lambda x: cloudsen12_style.get(x, x)

    # Download the model
    model = "https://github.com/IPL-UV/phicloudmask/releases/download/alpha/cloudmask_weights.pt"
    embedding = "https://github.com/IPL-UV/phicloudmask/releases/download/alpha/spectral_embedding.pt"

    model_path = pathlib.Path(outpath) / "cloudmask_weights.pt"
    embedding_path = pathlib.Path(outpath) / "spectral_embedding.pt"

    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(model, stream=True) as r:
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

    if not embedding_path.exists():
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(embedding, stream=True) as r:
            with open(embedding_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

    # Load the weights
    embedding_weights = torch.load(embedding_path)
    cloudmask_weights = torch.load(model_path)

    # Load the model
    model = CloudMask(descriptor=SENTINEL2_DESCRIPTORS, device=device)
    model.embedding_model.load_state_dict(embedding_weights)
    model.cloud_model.load_state_dict(cloudmask_weights)

    # Remove the gradient estimation
    for param in model.embedding_model.parameters():
        param.requires_grad = False

    for param in model.cloud_model.parameters():
        param.requires_grad = False

    # To eval model
    model = model.eval()
    model = model.to(device)

    return model


def run_cloud_mask(folder: pathlib.Path, device="cpu"):
    # Define the semantic categories mapping
    cloudsen12_style = {
        0: 0,
        1: 0,
        2: 0,
        6: 0,  # Merged into category 0 (land, water, snow, no_data)
        4: 1,  # Thick cloud -> category 1
        3: 2,  # Thin cloud -> category 2
        5: 3,  # Shadow -> category 3
    }
    map_values = lambda x: cloudsen12_style.get(x, x)

    folder_cloud = folder / "cloud_results"
    folder_cloud.mkdir(exist_ok=True)

    # Load the model
    cloud_mask02 = cloud_model2(device=device)
    cloud_mask03 = cloud_model3(device=device)
    geotiff_files = list((folder / "raw").glob("*.tif"))

    for idx, file in enumerate(geotiff_files):
        print(f"Processing {idx + 1}/{len(geotiff_files)}")
        with rio.open(file) as src:
            data = src.read()

        # Load the S2 data
        s2_data = torch.from_numpy(data[0:13])[None] / 10000
        s2_data = s2_data.to(device)

        # Run the cloud mask
        target = data[13, 128:384, 128:384]
        mask1 = data[14, 128:384, 128:384].astype(np.uint8)
        mask2 = cloud_mask02(s2_data).argmax(dim=1)[0, 128:384, 128:384].cpu().numpy()
        cloud_mask_all = cloud_mask03(s2_data).argmax(dim=0).cpu().numpy()
        mask3 = np.vectorize(map_values)(cloud_mask_all)[128:384, 128:384]

        # [0, 128:384, 128:384].cpu().numpy()
        s2_data = s2_data[0, 0:13, 128:384, 128:384].cpu().numpy()

        # Compare the masks with the target
        fbeta1 = fbeta_score(target.flatten() > 0, mask1.flatten() > 0, beta=2)
        fbeta2 = fbeta_score(target.flatten() > 0, mask2.flatten() > 0, beta=2)
        fbeta3 = fbeta_score(target.flatten() > 0, mask3.flatten() > 0, beta=2)

        # which is the best?
        best_model = np.argmax([fbeta1, fbeta2, fbeta3])
        model_name = ["model 1", "model 2", "model 3"]

        # Save the results
        dataset = {
            "s2_data": s2_data,
            "target": target,
            "mask1": mask1,
            "mask2": mask2,
            "mask3": mask3,
            "best_model": model_name[best_model],
            "metrics": [fbeta1, fbeta2, fbeta3],
            "name": file.stem,
        }

        with open(folder_cloud / f"{file.stem}.pkl", "wb") as f:
            pickle.dump(dataset, f)


# Create metadata
def select_hard_images(folder: pathlib.Path):
    list_f = list((folder / "cloud_results").glob("*.pkl"))
    list_f_container = []
    for f in list_f:
        with open(f, "rb") as file:
            dataf = pickle.load(file)
            best_model = dataf["best_model"]
            metrics01 = dataf["metrics"][0]
            metrics02 = dataf["metrics"][1]
            metrics03 = dataf["metrics"][2]
            name = dataf["name"]
            meta_dict = {
                "best_model": best_model.replace(" ", "_"),
                "f2score_model_01": metrics01,
                "f2score_model_02": metrics02,
                "f2score_model_03": metrics03,
                "name": name,
            }
            list_f_container.append(meta_dict)
    return pd.DataFrame(list_f_container)
