import os
import re
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import gdown
from torchvision.transforms.functional import to_pil_image
import torch
import torch.nn as nn
import torch.nn.functional as F

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣",
             "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
       'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

unique_chars = set(provinces[:-1] + alphabets[:-1] + ads[:-1])  # 'O' not included
char_list = sorted(list(unique_chars))
char_list = ["-"] + char_list
char2idx = {char: i for i, char in enumerate(char_list)}
idx2char = {i: c for c, i in char2idx.items()}
num_classes = len(char_list)


def decode_plate(s):
    idx   = list(map(int, s.split("_")))
    try:
        return provinces[idx[0]] + alphabets[idx[1]] + "".join(ads[i] for i in idx[2:])
    except Exception:
        return None

def split_bbox(bbox_str):
    # Split the string on one or more underscores
    tokens = re.split(r'_+', bbox_str)
    
    # Check if there are exactly 4 numeric tokens
    if len(tokens) == 4 and all(t.isdigit() for t in tokens):
        return tuple(map(int, tokens))
    
    # Return a tuple of Nones if the format is invalid
    return (None,) * 4

def crop_and_resize(img, x1, y1, x2, y2):
    # Check that the bounding box is valid
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Crop the image using the bounding box
    cropped_img = img[y1:y2, x1:x2]

    # Check that the cropped image is not empty
    if cropped_img.size == 0:
        return None

    # Resize the cropped image to 144x48 pixels
    try:
        return cv2.resize(cropped_img, (144, 48))
    except Exception as e:
        return None

def decode_ccpd_label(label_str, provinces, alphabets, ads):
    """
    Decodes a string like '0_0_22_27_27_33_16' into a license plate, e.g., '皖AWWX6G'
    """
    # Convert the underscore-separated string into a list of integers
    indices = list(map(int, label_str.strip().split('_')))
    
    # The label must contain exactly 7 indices
    if len(indices) != 7:
        raise ValueError("Label must contain 7 indices")

    # Decode each part of the license plate
    province = provinces[indices[0]]
    alphabet = alphabets[indices[1]]
    ad_chars = [ads[i] for i in indices[2:]]

    # Return the full license plate string
    return province + alphabet + ''.join(ad_chars)

def encode_plate(plate_str, char2idx):
    """
    Converts a license plate string like '皖AWWX6G'
    into a list of indices [3, 12, 30, 30, ...]
    """
    # Map each character to its corresponding index
    return [char2idx[c] for c in plate_str]

def decode_plate_from_list(label_indices, idx2char):
    """
    Converts a list of indices [3, 12, 30, ...] back into a 
    license plate string like '皖AWWX6G'
    """
    # Map each index back to its corresponding character
    return ''.join([idx2char[i] for i in label_indices])


def greedy_decode(logits, blank_index, idx2char):
    preds = logits.argmax(dim=2)  # (B, T)
    decoded_batch = []
    for pred in preds:
        chars = []
        prev = None
        for p in pred:
            p = p.item()
            if p != blank_index and p != prev:
                chars.append(idx2char[p])
            prev = p
        decoded_batch.append(''.join(chars))
    return decoded_batch


def create_dataframe(folder_path, char2idx):
    all_files = sorted(os.listdir(folder_path))
    jpg_files = [f for f in all_files if f.endswith('.jpg')]

    rows = []
    for fname in jpg_files:
        parts = fname[:-4].split("-")
        if len(parts) < 6:
            continue

        try:
            x1, y1, x2, y2 = split_bbox(parts[2])
            plate = decode_plate(parts[4])
            label = encode_plate(plate, char2idx)
        except Exception as e:
            print(f"Errore con file {fname}: {e}")
            continue

        rows.append({
            "image_path": os.path.join(folder_path, fname),
            "x1_bbox": x1, "y1_bbox": y1,
            "x2_bbox": x2, "y2_bbox": y2,
            "plate_number": plate,
            "label": label
        })

    return pd.DataFrame(rows)



def create_cropped_dataframe(df, cropped_folder):
    """
    Creates a new DataFrame with cropped and resized images if not already done.

    Args:
        df (pd.DataFrame): Original DataFrame containing bounding boxes and additional info.
        cropped_folder (str): Folder where cropped images will be saved.

    Returns:
        pd.DataFrame: New DataFrame with paths to cropped images, plate_number, and label.
    """
    
    os.makedirs(cropped_folder, exist_ok=True)
    
    # Verifica se il numero di file nella cartella corrisponde al numero di righe del DataFrame
    existing_files = [f for f in os.listdir(cropped_folder) if f.startswith("cropped_") and f.endswith(".jpg")]
    if len(existing_files) == len(df):
        # Se già croppate, crea direttamente il DataFrame dai percorsi esistenti
        cropped_rows = []
        for i, row in df.iterrows():
            cropped_path = os.path.join(cropped_folder, f"cropped_{i}.jpg")
            cropped_rows.append({
                "image_path": cropped_path,
                "plate_number": row["plate_number"],
                "label": row["label"]
            })
        return pd.DataFrame(cropped_rows)

    # Altrimenti, ricroppa tutte le immagini
    cropped_rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row["image_path"]
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found or corrupted: {image_path}")
            continue

        try:
            x1 = int(float(row["x1_bbox"]))
            y1 = int(float(row["y1_bbox"]))
            x2 = int(float(row["x2_bbox"]))
            y2 = int(float(row["y2_bbox"]))
        except Exception as e:
            print(f"Error parsing bounding box for {image_path}: {e}")
            continue

        resized_img = crop_and_resize(img, x1, y1, x2, y2)
        if resized_img is None:
            print(f"Error cropping/resizing image: {image_path}")
            continue

        cropped_path = os.path.join(cropped_folder, f"cropped_{i}.jpg")
        cv2.imwrite(cropped_path, resized_img)

        cropped_rows.append({
            "image_path": cropped_path,
            "plate_number": row["plate_number"],
            "label": row["label"]
        })

    return pd.DataFrame(cropped_rows)

def download_and_extract_dataset(url, output_path, extract_path, extracted_folder_path):
    """
    Downloads and extracts a dataset if not already present.

    Args:
        url (str): Google Drive URL of the dataset.
        output_path (str): Path where the .tar file will be saved.
        extract_path (str): Directory where the archive will be extracted.
        extracted_folder_path (str): Expected folder resulting from extraction.
    """
    
    # Download the dataset if it doesn't already exist
    if not os.path.exists(output_path):
        print("Downloading the dataset...")
        gdown.download(url, output_path, fuzzy=True, quiet=False)
    else:
        print("Dataset already exists, download skipped.")

    # Extract the dataset if the folder doesn't already exist
    if not os.path.exists(extracted_folder_path):
        print("Extracting the dataset...")
        os.makedirs(extract_path, exist_ok=True)
        with tarfile.open(output_path) as tar:
            tar.extractall(path=extract_path)
        print("Extraction completed.")
    else:
        print("Dataset folder already exists, extraction skipped.")

def plot_batch_images(train_loader, idx2char, font):
    images, labels = next(iter(train_loader))
    
    # 25 random index from the dataset
    indices = np.random.choice(len(images), size=25, replace=False)

    fig, axes = plt.subplots(5, 5, figsize=(20, 10))  
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        image = images[idx]
        label = labels[idx]

        decoded_plate = decode_plate_from_list([int(i) for i in label], idx2char)
        img_np = to_pil_image(image)

        ax.imshow(img_np)
        ax.set_title(f"Plate: {decoded_plate}", fontproperties=font, fontsize=18)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

