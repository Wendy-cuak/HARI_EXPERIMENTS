# HARI_EXPERIMENTS

- **dataset_generation.py**

1. We download the CloudSEN12 test dataset, which consists of Sentinel-2 images (13 bands) with human-labeled data.
2. We filter the images, selecting only those with more than 50% cloud coverage.
3. We run three popular cloud detection algorithms (UnetMobV2 v1, UnetMobV2 v2, and SenSeiV2).
4. We identify the most challenging images based on the F2-score, then create "pickle" files containing the following information:
    - s2_data: The original Sentinel-2 image data.
    - target: The ground truth labels for cloud detection.
    - mask1: The mask produced by UnetMobV2 v1.
    - mask2: The mask produced by UnetMobV2 v2.
    - mask3: The mask produced by SenSeiV2.
    - best_model: The cloud detection model that achieved the best F2-score for the image.
    - metrics: Performance F2-score for each model.
    - name: The image name.


- **display.py**
  
The display script permits a fast view of the pickle information.

![image](https://github.com/user-attachments/assets/3d3e26e1-ea5c-4ef1-837b-5bae73ab63e9)

- **pickles files**

Results after run dataset_generation.py

[dataset_folder](https://drive.google.com/drive/folders/10X1aSppbKKTvbqOlGEkJky21L9AWAmyE?usp=drive_link)
