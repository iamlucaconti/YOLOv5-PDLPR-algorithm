# YOLOv5 Fine-Tuning and Evaluation on CCPD

The notebook yolov5 is used for finetuning phase and test phase.

---

## Dataset Overview

We used:

- **50,000 images** from the **CCPD-Base** dataset for training
  - Split: **80% train**, **20% val**

For evaluation, we tested on **8 subsets**, each with **1,000 images**, totaling **8,000 test samples**, covering different distortions and challenges:

| Sub-Dataset       | Description                                                                                  | # Images |
|-------------------|----------------------------------------------------------------------------------------------|----------|
| **CCPD-Base**     | Standard license plate images under ideal conditions.                                        | 1,000    |
| **CCPD-FN**       | Plates that are either very close or far from the camera.                                    | 1,000    |
| **CCPD-DB**       | Plates with extreme lighting (brighter/darker/uneven).                                       | 1,000    |
| **CCPD-Rotate**   | Plates tilted **20–50° horizontally**, and **–10 to 10° vertically**.                        | 1,000    |
| **CCPD-Tilt**     | Plates tilted **15–45° horizontally and vertically**.                                        | 1,000    |
| **CCPD-Weather**  | Plates photographed in **rain, snow, or fog**.                                               | 1,000    |
| **CCPD-Challenge**| The most difficult images in the dataset.                                                    | 1,000    |
| **CCPD-Blur**     | Blurred plates due to motion or camera shake.                                                | 1,000    |
| **CCPD-NP**       | Images of new vehicles **without license plates** (used to evaluate false positive rate).    | 1,000    |


## Test Phase

We evaluated the fine-tuned model separately on each test subset using `val.py` from YOLOv5.
The results are in "test_results" folder, each containg example of prediction and metric graphics.
Also we computed other metrics such as: "Recall_IoU>0.7", "Precision_IoU>0.7", "Accuracy_img_IoU>0.7" (metric_iou0.7.csv).
The fps is computed in the notebook yolov5, the value is ≈ 95 fps (all_imgs/all_seconds) using a P100 gpu on Kaggle environment.