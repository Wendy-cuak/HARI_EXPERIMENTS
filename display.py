import pathlib
import pickle
import pandas as pd

DATASET_FOLDER = pathlib.Path("dataset")
TO_DISPLAY = 50


# Display the image
dataset = pd.read_csv(DATASET_FOLDER / "hard_images.csv")
files = "dataset/cloud_results/" + dataset["name"] + ".pkl"

with open(files[TO_DISPLAY], "rb") as file:
    data = pickle.load(file)

# Display the image
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 5, figsize=(25, 5))
ax[0].imshow(np.moveaxis(data["s2_data"], 0, -1)[..., [3, 2, 1]] * 2)
ax[0].set_title("S2 image")
ax[1].imshow(data["target"])
ax[1].set_title("Ground truth")
ax[2].imshow(data["mask1"])
ax[2].set_title("Mask 1")
ax[3].imshow(data["mask2"])
ax[3].set_title("Mask 2")
ax[4].imshow(data["mask3"])
ax[4].set_title("Mask 3")
plt.show()

print(f"Best model: {data['best_model']}")
